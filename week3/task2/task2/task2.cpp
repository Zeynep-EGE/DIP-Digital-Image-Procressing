#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

// 1) 256 degerli histogram
std::vector<int> histogramCikar(const cv::Mat& img)
{
    std::vector<int> hist(256, 0);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            hist[img.at<uchar>(i, j)]++;
        }
    }
    return hist;
}

// 2) Kumulatif dagilim (CDF)
std::vector<int> cdfHesapla(const std::vector<int>& hist)
{
    std::vector<int> cdf(256, 0);
    cdf[0] = hist[0];
    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i - 1] + hist[i];
    }
    return cdf;
}

// 3) Histogram esitleme
cv::Mat histogramEsitle(const cv::Mat& src, const std::vector<int>& cdf)
{
    int toplam = src.rows * src.cols;

    // CDF'deki sifirdan buyuk ilk deger
    int cdfMin = 0;
    for (int i = 0; i < 256; i++) {
        if (cdf[i] > 0) { cdfMin = cdf[i]; break; }
    }

    cv::Mat dst = src.clone();

    // Tek renkli resimde bolme hatasi olmasin
    if (toplam == cdfMin) return dst;

    // Donusum tablosu (look-up table)
    std::vector<uchar> lut(256, 0);
    for (int i = 0; i < 256; i++) {
        double deger = (double)(cdf[i] - cdfMin) / (toplam - cdfMin) * 255.0;
        if (deger < 0) deger = 0;      // cdf < cdfMin olan (bos) seviyeler icin
        lut[i] = static_cast<uchar>(deger + 0.5);   // yuvarla
    }

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            dst.at<uchar>(i, j) = lut[src.at<uchar>(i, j)];
        }
    }
    return dst;
}

int main()
{
    // Tek kanalli (gri) olarak ac
    cv::Mat img = cv::imread("C:/Users/Kadir/Desktop/DIP-Digital-Image-Procressing/images/test2.jpeg", cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cout << "Resim okunamadi! Dosya yolunu kontrol et." << std::endl;
        return -1;
    }

    std::vector<int> hist = histogramCikar(img);
    std::vector<int> cdf = cdfHesapla(hist);
    cv::Mat sonuc = histogramEsitle(img, cdf);

    // Kontrol: CDF'nin son elemani toplam piksel sayisina esit olmali
    std::cout << "Toplam piksel: " << img.rows * img.cols
        << "  CDF son deger: " << cdf[255] << std::endl;

    // Esitlenmis resmin histogramini da cikarip yazdir (istege bagli)
    std::vector<int> hist2 = histogramCikar(sonuc);
    std::cout << "Deger\tOnceki\tSonraki" << std::endl;
    for (int i = 0; i < 256; i++) {
        std::cout << i << "\t" << hist[i] << "\t" << hist2[i] << std::endl;
    }

    cv::imshow("Orijinal", img);
    cv::imshow("Histogram Esitleme", sonuc);
    cv::imwrite("esitlenmis.jpg", sonuc);

    cv::waitKey(0);
    return 0;
}