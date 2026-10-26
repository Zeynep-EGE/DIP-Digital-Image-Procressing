#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>

//Bu görevleri yapay zeka yardımı ile yaptımm

static cv::Mat logKernel(int boyut, double sigma)
{
    cv::Mat k(boyut, boyut, CV_32F);
    int r = boyut / 2;
    double s2 = sigma * sigma;

    for (int y = -r; y <= r; y++) {
        for (int x = -r; x <= r; x++) {
            double d = (x * x + y * y) / (2.0 * s2);
            k.at<float>(y + r, x + r) =
                (float)(-1.0 / (CV_PI * s2 * s2) * (1.0 - d) * std::exp(-d));
        }
    }

    
    k -= cv::mean(k)[0];
    return k;
}

static cv::Mat gosterim(const cv::Mat& m32)
{
    cv::Mat a = cv::abs(m32), out;
    cv::normalize(a, a, 0, 255, cv::NORM_MINMAX);
    a.convertTo(out, CV_8U);
    return out;
}

static cv::Mat sobelBuyukluk(const cv::Mat& img)
{
    cv::Mat kx = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat ky = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    cv::Mat gx, gy, m;
    cv::filter2D(img, gx, CV_32F, kx);
    cv::filter2D(img, gy, CV_32F, ky);
    cv::magnitude(gx, gy, m);
    return m;
}

static cv::Mat prewittBuyukluk(const cv::Mat& img)
{
    cv::Mat kx = (cv::Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
    cv::Mat ky = (cv::Mat_<float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
    cv::Mat gx, gy, m;
    cv::filter2D(img, gx, CV_32F, kx);
    cv::filter2D(img, gy, CV_32F, ky);
    cv::magnitude(gx, gy, m);
    return m;
}

static cv::Mat laplacian3x3(const cv::Mat& img)
{
    cv::Mat k = (cv::Mat_<float>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
    cv::Mat out;
    cv::filter2D(img, out, CV_32F, k);
    return out;
}

static cv::Mat logFiltre(const cv::Mat& img, const cv::Mat& k)
{
    cv::Mat out;
    cv::filter2D(img, out, CV_32F, k);
    return out;
}

static cv::Mat etiketle(const cv::Mat& g, const std::string& yazi)
{
    cv::Mat k = g.clone();
    cv::putText(k, yazi, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6,
        cv::Scalar(255), 3);
    cv::putText(k, yazi, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6,
        cv::Scalar(0), 1);
    return k;
}

int main(int argc, char** argv)
{
    std::string yol = (argc > 1) ? argv[1] : "C:/Users/Kadir/Desktop/DIP-Digital-Image-Procressing/images/lenna.png";
    double sigma = (argc > 2) ? std::atof(argv[2]) : 1.0;

    cv::Mat gri = cv::imread(yol, cv::IMREAD_GRAYSCALE);
    if (gri.empty()) {
        std::cerr << "Görüntü okunamadı: " << yol << std::endl;
        return -1;
    }

    cv::Mat yumusak;
    cv::GaussianBlur(gri, yumusak, cv::Size(3, 3), 1.0);

    cv::Mat log7 = logKernel(7, sigma);
    std::cout << "7x7 LoG cekirdegi (sigma = " << sigma << "):\n" << log7 << std::endl;

    cv::Mat sobelHam = gosterim(sobelBuyukluk(gri));
    cv::Mat prewittHam = gosterim(prewittBuyukluk(gri));
    cv::Mat lapHam = gosterim(laplacian3x3(gri));
    cv::Mat logHam = gosterim(logFiltre(gri, log7));

    cv::Mat sobelGauss = gosterim(sobelBuyukluk(yumusak));
    cv::Mat prewittGauss = gosterim(prewittBuyukluk(yumusak));
    cv::Mat lapGauss = gosterim(laplacian3x3(yumusak));
    cv::Mat logGauss = gosterim(logFiltre(yumusak, log7));

    cv::imwrite("sobel_gaussiz.png", sobelHam);
    cv::imwrite("prewitt_gaussiz.png", prewittHam);
    cv::imwrite("laplacian_gaussiz.png", lapHam);
    cv::imwrite("log7_gaussiz.png", logHam);
    cv::imwrite("sobel_gaussli.png", sobelGauss);
    cv::imwrite("prewitt_gaussli.png", prewittGauss);
    cv::imwrite("laplacian_gaussli.png", lapGauss);
    cv::imwrite("log7_gaussli.png", logGauss);

    auto kucult = [](const cv::Mat& m) {
        cv::Mat o;
        double oran = 320.0 / m.cols;
        cv::resize(m, o, cv::Size(), oran, oran, cv::INTER_AREA);
        return o;
        };

    cv::Mat ust, alt;
    cv::hconcat(std::vector<cv::Mat>{
        etiketle(kucult(sobelHam), "Sobel"),
            etiketle(kucult(prewittHam), "Prewitt"),
            etiketle(kucult(lapHam), "Laplacian"),
            etiketle(kucult(logHam), "LoG 7x7")}, ust);
    cv::hconcat(std::vector<cv::Mat>{
        etiketle(kucult(sobelGauss), "Sobel+Gauss"),
            etiketle(kucult(prewittGauss), "Prewitt+Gauss"),
            etiketle(kucult(lapGauss), "Laplacian+Gauss"),
            etiketle(kucult(logGauss), "LoG+Gauss")}, alt);

    cv::Mat montaj;
    cv::vconcat(ust, alt, montaj);
    cv::imwrite("karsilastirma_montaj.png", montaj);

    cv::imshow("Orijinal (gri)", gri);
    cv::imshow("Karsilastirma (ust: Gauss'suz, alt: Gauss'lu)", montaj);
    cv::waitKey(0);
    return 0;
}