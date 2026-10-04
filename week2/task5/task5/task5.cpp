#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <array>
#include <chrono>

void histogramCikar(const cv::Mat& roi, std::array<int, 256>* hist)
{
    hist->fill(0);

    for (int i = 0; i < roi.rows; i++) {
        const uchar* satir = roi.ptr<uchar>(i);   
        for (int j = 0; j < roi.cols; j++) {
            (*hist)[satir[j]]++;
        }
    }
}

cv::Mat histogramCiz(const std::array<int, 256>& hist, const std::string& baslik)
{
    const int genislik = 512, yukseklik = 400;
    cv::Mat img(yukseklik, genislik, CV_8UC3, cv::Scalar(255, 255, 255));

    int enBuyuk = 1;
    for (int v : hist) if (v > enBuyuk) enBuyuk = v;

    int kolon = genislik / 256;  
    for (int v = 0; v < 256; v++) {
        int h = (int)((double)hist[v] / enBuyuk * (yukseklik - 40));
        cv::rectangle(img,
            cv::Point(v * kolon, yukseklik - 1),
            cv::Point(v * kolon + kolon - 1, yukseklik - 1 - h),
            cv::Scalar(60, 60, 60), cv::FILLED);
    }
    cv::putText(img, baslik, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 200), 2);
    return img;
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

    int n = std::thread::hardware_concurrency();
    if (n <= 0) n = 4;
    std::cout << "Parca / thread sayisi (n): " << n << std::endl;

    // ---------- 1) Tek çekirdek (karşılaştırma için) ----------
    std::array<int, 256> histTek;

    auto t1 = std::chrono::high_resolution_clock::now();
    histogramCikar(gray, &histTek);
    auto t2 = std::chrono::high_resolution_clock::now();
    double sureTek = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // ---------- 2) n parça, n thread, n ayrı histogram ----------
    // Onceden n elemanlik boyutlandiriyoruz, boylece pointerlar gecerli kalir
    std::vector<std::array<int, 256>> parcaHist(n);
    std::vector<std::thread> threadler;

    int parcaYuksek = gray.rows / n;

    auto t3 = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < n; k++) {
        int y0 = k * parcaYuksek;
        int h = (k == n - 1) ? (gray.rows - y0) : parcaYuksek;

        cv::Rect bolge(0, y0, gray.cols, h);
        cv::Mat roi = gray(bolge);   // kopya degil, pencere

        threadler.emplace_back(histogramCikar, roi, &parcaHist[k]);
    }

    for (auto& t : threadler) t.join();

    // ---------- 3) Parça histogramlarını topla ----------
    std::array<int, 256> histToplam;
    histToplam.fill(0);

    for (int k = 0; k < n; k++) {
        for (int v = 0; v < 256; v++) {
            histToplam[v] += parcaHist[k][v];
        }
    }

    auto t4 = std::chrono::high_resolution_clock::now();
    double sureCoklu = std::chrono::duration<double, std::milli>(t4 - t3).count();

    // ---------- Doğrulama ----------
    long long toplamPiksel = 0;
    int farkliKutu = 0;
    for (int v = 0; v < 256; v++) {
        toplamPiksel += histToplam[v];
        if (histToplam[v] != histTek[v]) farkliKutu++;
    }

    std::cout << "Tek cekirdek : " << sureTek << " ms" << std::endl;
    std::cout << n << " cekirdek   : " << sureCoklu << " ms" << std::endl;
    std::cout << "Histogram toplam piksel : " << toplamPiksel
        << " (resim: " << (long long)gray.rows * gray.cols << ")" << std::endl;
    std::cout << "Tek cekirdekle farkli kutu sayisi: " << farkliKutu << " (0 olmali)" << std::endl;

    // OpenCV'nin kendi calcHist'i ile de karsilastir
    cv::Mat cvHist;
    int kutu = 256;
    float aralik[] = { 0, 256 };
    const float* aralikPtr = aralik;
    cv::calcHist(&gray, 1, 0, cv::Mat(), cvHist, 1, &kutu, &aralikPtr);

    int cvFark = 0;
    for (int v = 0; v < 256; v++) {
        if ((int)cvHist.at<float>(v) != histToplam[v]) cvFark++;
    }
    std::cout << "cv::calcHist ile farkli kutu sayisi: " << cvFark << " (0 olmali)" << std::endl;

    // Ilk 10 parlaklik degerini yazdir
    std::cout << "\nParlaklik -> Adet (ilk 10)" << std::endl;
    for (int v = 0; v < 10; v++) {
        std::cout << v << " -> " << histToplam[v] << std::endl;
    }

    // ---------- Görselleştirme ----------
    cv::Mat cizim = histogramCiz(histToplam, "Toplam histogram");
    cv::imwrite("histogram_toplam.png", cizim);

    // Her parçanın histogramı da ayrı pencerede
    for (int k = 0; k < n; k++) {
        std::string ad = "Parca " + std::to_string(k + 1);
        cv::imshow(ad, histogramCiz(parcaHist[k], ad));
    }

    cv::namedWindow("Orijinal gri", cv::WINDOW_NORMAL);
    cv::imshow("Orijinal gri", gray);
    cv::imshow("Toplam histogram", cizim);

    cv::waitKey(0);
    return 0;
}