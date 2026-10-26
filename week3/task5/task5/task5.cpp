#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------
// Test icin tuz-biber gurultusu ekleme
// oran: 0.10 = piksellerin %10'u bozulur (yarisi 0, yarisi 255)
// ---------------------------------------------------------
cv::Mat tuzBiberEkle(const cv::Mat& src, double oran)
{
    cv::Mat dst = src.clone();
    std::mt19937 gen(42);                       // sabit tohum: sonuc tekrarlanabilir
    std::uniform_real_distribution<double> u(0.0, 1.0);

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            double r = u(gen);
            if (r < oran / 2)  dst.at<uchar>(i, j) = 0;      // biber
            else if (r < oran) dst.at<uchar>(i, j) = 255;    // tuz
        }
    }
    return dst;
}

// ---------------------------------------------------------
// 5x5 medyan filtre (hazir fonksiyon yok)
// ---------------------------------------------------------
cv::Mat medyanFiltre5x5(const cv::Mat& src)
{
    CV_Assert(src.type() == CV_8UC1);

    const int R = 2;                 // 5x5 icin yaricap 2
    const int N = 25;                // pencerede 25 eleman var
    cv::Mat dst(src.rows, src.cols, CV_8UC1);

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {

            // 1) 5x5 penceredeki 25 degeri diziye topla
            uchar w[N];
            int k = 0;
            for (int m = -R; m <= R; m++) {
                for (int n = -R; n <= R; n++) {
                    // Kenarda resmin disina tasarsa en yakin kenar pikselini kullan
                    int y = std::max(0, std::min(src.rows - 1, i + m));
                    int x = std::max(0, std::min(src.cols - 1, j + n));
                    w[k++] = src.at<uchar>(y, x);
                }
            }

            // 2) Diziyi kucukten buyuge sirala (insertion sort, elle)
            for (int a = 1; a < N; a++) {
                uchar anahtar = w[a];
                int b = a - 1;
                while (b >= 0 && w[b] > anahtar) {
                    w[b + 1] = w[b];
                    b--;
                }
                w[b + 1] = anahtar;
            }

            // 3) Ortadaki (13.) eleman = medyan
            dst.at<uchar>(i, j) = w[N / 2];
        }
    }
    return dst;
}

// ---------------------------------------------------------
// PSNR: iki resmin benzerligi (yuksek = daha iyi)
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
    // Temiz resmi oku ve uzerine tuz-biber ekle
    cv::Mat temiz = cv::imread("C:/Users/Kadir/Desktop/DIP-Digital-Image-Procressing/images/lenna.png", cv::IMREAD_GRAYSCALE);
    if (temiz.empty()) {
        std::cout << "Resim okunamadi!" << std::endl;
        return -1;
    }
    cv::Mat gurultulu = tuzBiberEkle(temiz, 0.10);

    // Elinde hazir tuz-biber gurultulu resim varsa yukaridaki iki satir yerine:
    // cv::Mat gurultulu = cv::imread("gurultulu.png", cv::IMREAD_GRAYSCALE);
    // (bu durumda PSNR hesaplari yapilamaz, cunku temiz resim yok)

    // 1) HAZIR OpenCV medyan filtre (5x5)
    cv::Mat hazir;
    cv::medianBlur(gurultulu, hazir, 5);

    // 2) KENDI medyan filtremiz (5x5)
    cv::Mat elle = medyanFiltre5x5(gurultulu);

    // 3) TEST A: iki sonuc ayni mi?
    int maxFark = 0;
    long long farkliPiksel = 0;
    for (int i = 0; i < elle.rows; i++) {
        for (int j = 0; j < elle.cols; j++) {
            int fark = std::abs((int)elle.at<uchar>(i, j) - (int)hazir.at<uchar>(i, j));
            if (fark > maxFark) maxFark = fark;
            if (fark != 0) farkliPiksel++;
        }
    }
    std::cout << "Hazir ile elle arasindaki maksimum fark: " << maxFark << std::endl;
    std::cout << "Farkli piksel sayisi                   : " << farkliPiksel << std::endl;

    // 4) TEST B: gurultu gercekten azaldi mi? (temiz resme gore PSNR)
    cv::Mat ortalama;
    cv::blur(gurultulu, ortalama, cv::Size(5, 5), cv::Point(-1, -1), cv::BORDER_REPLICATE);

    std::cout << "PSNR gurultulu resim   : " << psnrHesapla(temiz, gurultulu) << " dB" << std::endl;
    std::cout << "PSNR 5x5 medyan (elle) : " << psnrHesapla(temiz, elle) << " dB" << std::endl;
    std::cout << "PSNR 5x5 ortalama      : " << psnrHesapla(temiz, ortalama) << " dB" << std::endl;

    cv::imwrite("gurultulu_tuzbiber.png", gurultulu);
    cv::imwrite("medyan_hazir.png", hazir);
    cv::imwrite("medyan_elle.png", elle);

    cv::imshow("Gurultulu", gurultulu);
    cv::imshow("Medyan - Hazir", hazir);
    cv::imshow("Medyan - Elle", elle);
    cv::imshow("Ortalama 5x5 (karsilastirma)", ortalama);
    cv::waitKey(0);
    return 0;
}