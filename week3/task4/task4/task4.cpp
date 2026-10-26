#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------
// Test icin gurultu ekleme fonksiyonlari
// ---------------------------------------------------------
cv::Mat gaussGurultuEkle(const cv::Mat& src, double sigma)
{
    cv::Mat dst = src.clone();
    std::mt19937 gen(42);                                   // sabit tohum: sonuc tekrarlanabilir
    std::normal_distribution<double> dagilim(0.0, sigma);

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            double v = src.at<uchar>(i, j) + dagilim(gen);
            v = std::max(0.0, std::min(255.0, v));
            dst.at<uchar>(i, j) = (uchar)(v + 0.5);
        }
    }
    return dst;
}

cv::Mat tuzBiberEkle(const cv::Mat& src, double oran)      // oran: 0.05 = %5 piksel bozulur
{
    cv::Mat dst = src.clone();
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> u(0.0, 1.0);

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            double r = u(gen);
            if (r < oran / 2)      dst.at<uchar>(i, j) = 0;     // biber (siyah)
            else if (r < oran)     dst.at<uchar>(i, j) = 255;   // tuz (beyaz)
        }
    }
    return dst;
}

// ---------------------------------------------------------
// Konvolusyon: herhangi boyutta (tek sayili) cekirdekle calisir
// Kenarlarda en yakin piksel tekrarlanir (replicate)
// ---------------------------------------------------------
cv::Mat konvolusyon(const cv::Mat& src, const std::vector<std::vector<double>>& k)
{
    CV_Assert(src.type() == CV_8UC1);

    int kr = (int)k.size() / 2;        // cekirdek yaricapi (3x3 icin 1)
    int kc = (int)k[0].size() / 2;

    cv::Mat dst(src.rows, src.cols, CV_8UC1);

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {

            double toplam = 0.0;

            for (int m = -kr; m <= kr; m++) {
                for (int n = -kc; n <= kc; n++) {
                    // Komsu piksel resmin disina tasarsa kenara sabitle
                    int y = std::max(0, std::min(src.rows - 1, i + m));
                    int x = std::max(0, std::min(src.cols - 1, j + n));

                    // Gercek konvolusyonda cekirdek ters cevrilir
                    toplam += src.at<uchar>(y, x) * k[kr - m][kc - n];
                }
            }

            toplam = std::max(0.0, std::min(255.0, toplam));
            dst.at<uchar>(i, j) = (uchar)(toplam + 0.5);
        }
    }
    return dst;
}

// ---------------------------------------------------------
// Kalite olcusu: PSNR (yuksek = iki resim birbirine daha yakin)
// ---------------------------------------------------------
double psnrHesapla(const cv::Mat& a, const cv::Mat& b)
{
    double mse = 0.0;
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            double d = (double)a.at<uchar>(i, j) - (double)b.at<uchar>(i, j);
            mse += d * d;
        }
    }
    mse /= (double)(a.rows * a.cols);
    if (mse == 0) return 99.0;
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}

// ---------------------------------------------------------
int main()
{
    cv::Mat temiz = cv::imread("C:/Users/Kadir/Desktop/DIP-Digital-Image-Procressing/images/lenna.png", cv::IMREAD_GRAYSCALE);
    if (temiz.empty()) {
        std::cout << "Resim okunamadi!" << std::endl;
        return -1;
    }

    // 1) Gurultulu resim olustur (kendi gurultulu resmin varsa bu satiri
    //    atlayip onu dogrudan 'gurultulu' olarak oku)
    cv::Mat gurultulu = gaussGurultuEkle(temiz, 20.0);
    // cv::Mat gurultulu = tuzBiberEkle(temiz, 0.05);

    // 2) 3x3 ortalama cekirdegi
    std::vector<std::vector<double>> ortalama(3, std::vector<double>(3, 1.0 / 9.0));

    // 3) Konvolusyonla filtrele
    cv::Mat filtreli = konvolusyon(gurultulu, ortalama);

    // 4) TEST A: hazir cv::blur ile karsilastir (sadece kontrol icin)
    cv::Mat hazir;
    cv::blur(gurultulu, hazir, cv::Size(3, 3), cv::Point(-1, -1), cv::BORDER_REPLICATE);

    int maxFark = 0;
    for (int i = 0; i < filtreli.rows; i++)
        for (int j = 0; j < filtreli.cols; j++)
            maxFark = std::max(maxFark,
                std::abs((int)filtreli.at<uchar>(i, j) - (int)hazir.at<uchar>(i, j)));

    std::cout << "cv::blur ile maksimum fark: " << maxFark << std::endl;

    // 5) TEST B: gurultu gercekten azaldi mi? (temiz resme gore PSNR)
    std::cout << "PSNR gurultulu resim : " << psnrHesapla(temiz, gurultulu) << " dB" << std::endl;
    std::cout << "PSNR filtreli resim  : " << psnrHesapla(temiz, filtreli) << " dB" << std::endl;

    cv::imwrite("gurultulu.png", gurultulu);
    cv::imwrite("filtreli.png", filtreli);

    cv::imshow("Temiz", temiz);
    cv::imshow("Gurultulu", gurultulu);
    cv::imshow("3x3 Ortalama Filtre", filtreli);
    cv::waitKey(0);
    return 0;
}