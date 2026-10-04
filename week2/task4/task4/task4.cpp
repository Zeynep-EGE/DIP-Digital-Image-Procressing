#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <chrono>

// Tek bir parçaya (ROI) kuantizasyon uygular
// dst değer olarak alınır: cv::Mat kopyalanınca piksel verisi kopyalanmaz,
// aynı belleğe yazılmaya devam edilir.
void kuantizeEt(const cv::Mat& src, cv::Mat dst, int L)
{
    try {
        int aralik = 256 / L;

        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                int piksel = src.at<uchar>(i, j);
                int basamak = piksel / aralik;
                dst.at<uchar>(i, j) = (uchar)(basamak * 255 / (L - 1));
            }
        }
    }
    catch (const cv::Exception& e) {
        std::cerr << "Thread hatasi: " << e.what() << std::endl;
    }
}

int main()
{
    const std::string yol = "C:/Users/Kadir/Desktop/DIP-Digital-Image-Procressing/images/test.jpeg";

    cv::Mat gray = cv::imread(yol, cv::IMREAD_GRAYSCALE);

    if (gray.empty()) {
        std::cerr << "Hata: Resim okunamadi! Aranan yol: " << yol << std::endl;
        std::cin.get();
        return -1;
    }

    std::cout << "Resim boyutu: " << gray.cols << "x" << gray.rows << std::endl;

    int n = std::thread::hardware_concurrency();  // mantiksal cekirdek sayisi
    if (n <= 0) n = 4;
    std::cout << "Parca / thread sayisi (n): " << n << std::endl;

    const int L = 8;  // ton sayisi

    // ---------- 1) Tek çekirdek (karşılaştırma için) ----------
    cv::Mat tek = cv::Mat::zeros(gray.size(), CV_8UC1);

    auto t1 = std::chrono::high_resolution_clock::now();
    kuantizeEt(gray, tek, L);
    auto t2 = std::chrono::high_resolution_clock::now();

    double sureTek = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // ---------- 2) n çekirdek, n ROI ----------
    cv::Mat coklu = cv::Mat::zeros(gray.size(), CV_8UC1);

    int parcaYuksek = gray.rows / n;
    std::vector<std::thread> threadler;

    auto t3 = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < n; k++) {
        int y0 = k * parcaYuksek;
        // Son parça, bölümden artan satırları da alır
        int h = (k == n - 1) ? (gray.rows - y0) : parcaYuksek;

        cv::Rect bolge(0, y0, gray.cols, h);

        cv::Mat srcROI = gray(bolge);    // kopya DEGIL, pencere
        cv::Mat dstROI = coklu(bolge);

        threadler.emplace_back(kuantizeEt, srcROI, dstROI, L);
    }

    for (auto& t : threadler) t.join();   // hepsinin bitmesini bekle

    auto t4 = std::chrono::high_resolution_clock::now();
    double sureCoklu = std::chrono::duration<double, std::milli>(t4 - t3).count();

    // ---------- Sonuçlar ----------
    std::cout << "Tek cekirdek : " << sureTek << " ms" << std::endl;
    std::cout << n << " cekirdek   : " << sureCoklu << " ms" << std::endl;

    // İki sonuç aynı mı?
    cv::Mat fark;
    cv::absdiff(tek, coklu, fark);
    std::cout << "Fark olan piksel sayisi: " << cv::countNonZero(fark) << " (0 olmali)" << std::endl;

    cv::imwrite("quant_tek.png", tek);
    cv::imwrite("quant_coklu.png", coklu);

    cv::namedWindow("Orijinal gri", cv::WINDOW_NORMAL);
    cv::namedWindow("Coklu cekirdek sonucu", cv::WINDOW_NORMAL);
    cv::imshow("Orijinal gri", gray);
    cv::imshow("Coklu cekirdek sonucu", coklu);

    cv::waitKey(0);
    return 0;
}