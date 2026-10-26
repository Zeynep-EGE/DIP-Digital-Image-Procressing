#include <opencv2/opencv.hpp>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

//Bunu yapay zeka ile yazdım 


// 7x7 Laplacian of Gaussian çekirdeği
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

static cv::Mat LOG7;  

//hafif Gauss yumuşatma
static cv::Mat gaussla(const cv::Mat& g)
{
    cv::Mat y;
    cv::GaussianBlur(g, y, cv::Size(3, 3), 1.0);
    return y;
}


static cv::Mat gosterim(const cv::Mat& m32)
{
    cv::Mat a = cv::abs(m32), out;
    cv::normalize(a, a, 0, 255, cv::NORM_MINMAX);
    a.convertTo(out, CV_8U);
    return out;
}

static cv::Mat esikle(const cv::Mat& harita, int t)
{
    cv::Mat o;
    cv::threshold(harita, o, t, 255, cv::THRESH_BINARY);
    return o;
}

static cv::Mat buyukluk(const cv::Mat& img, const cv::Mat& kx, const cv::Mat& ky)
{
    cv::Mat gx, gy, m;
    cv::filter2D(img, gx, CV_32F, kx);
    cv::filter2D(img, gy, CV_32F, ky);
    cv::magnitude(gx, gy, m);
    return m;
}


static void sentetikUret(cv::Mat& gri, cv::Mat& gt)
{
    const int N = 512;
    cv::Mat temiz(N, N, CV_8U);
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++)
            temiz.at<uchar>(y, x) = (uchar)(60 + x * 40 / N);  // yumuşak arka plan

    gt = cv::Mat::zeros(N, N, CV_8U);

    // Şekilleri hem görüntüye (dolu) hem gerçek kenar haritasına (çizgi) çiz
    cv::rectangle(temiz, cv::Point(80, 80), cv::Point(230, 200), cv::Scalar(200), cv::FILLED);
    cv::rectangle(gt, cv::Point(80, 80), cv::Point(230, 200), cv::Scalar(255), 1);

    cv::circle(temiz, cv::Point(360, 150), 70, cv::Scalar(30), cv::FILLED);
    cv::circle(gt, cv::Point(360, 150), 70, cv::Scalar(255), 1);

    cv::rectangle(temiz, cv::Point(100, 300), cv::Point(250, 440), cv::Scalar(140), cv::FILLED);
    cv::rectangle(gt, cv::Point(100, 300), cv::Point(250, 440), cv::Scalar(255), 1);

    cv::circle(temiz, cv::Point(380, 380), 60, cv::Scalar(220), cv::FILLED);
    cv::circle(gt, cv::Point(380, 380), 60, cv::Scalar(255), 1);

    std::vector<cv::Point> ucgen = { {300, 270}, {340, 330}, {260, 330} };
    cv::fillConvexPoly(temiz, ucgen, cv::Scalar(100));
    cv::polylines(gt, ucgen, true, cv::Scalar(255), 1);

    // Gauss gürültüsü 
    cv::setRNGSeed(42);
    cv::Mat gurultu(N, N, CV_32F), t32;
    cv::randn(gurultu, 0, 12);
    temiz.convertTo(t32, CV_32F);
    t32 += gurultu;
    t32.convertTo(gri, CV_8U);  
}

//Doğruluk Ölçümü
static void olc(const cv::Mat& tahmin, const cv::Mat& gt, int tol,
    double& P, double& R, double& F1)
{
    cv::Mat el = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * tol + 1, 2 * tol + 1));
    cv::Mat gtD, tD, a, b;
    cv::dilate(gt, gtD, el);
    cv::dilate(tahmin, tD, el);
    cv::bitwise_and(tahmin, gtD, a);  
    cv::bitwise_and(gt, tD, b);       

    double tahminSay = cv::countNonZero(tahmin);
    double gtSay = cv::countNonZero(gt);
    P = tahminSay > 0 ? cv::countNonZero(a) / tahminSay : 0.0;
    R = gtSay > 0 ? cv::countNonZero(b) / gtSay : 0.0;
    F1 = (P + R) > 0 ? 2 * P * R / (P + R) : 0.0;
}

// Süre ölçümü 
static double sureOlc(const std::function<cv::Mat(const cv::Mat&, int)>& f,
    const cv::Mat& gri, int t, int tekrar = 50)
{
    f(gri, t);  
    int64 bas = cv::getTickCount();
    for (int i = 0; i < tekrar; i++) f(gri, t);
    return (cv::getTickCount() - bas) / cv::getTickFrequency() * 1000.0 / tekrar;
}

static cv::Mat etiketle(const cv::Mat& g, const std::string& yazi)
{
    cv::Mat k = g.clone();
    cv::putText(k, yazi, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255), 3);
    cv::putText(k, yazi, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0), 1);
    return k;
}

struct Yontem {
    std::string ad;
    std::function<cv::Mat(const cv::Mat&, int)> kenar;  
};

int main(int argc, char** argv)
{
   
    cv::Mat gri, gt;
    bool gtVar = false;

    if (argc > 1) {
        gri = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
        if (gri.empty()) {
            std::cerr << "Görüntü okunamadı: " << argv[1] << std::endl;
            return -1;
        }
        if (argc > 2) {
            gt = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
            if (!gt.empty()) {
                if (gt.size() != gri.size())
                    cv::resize(gt, gt, gri.size(), 0, 0, cv::INTER_NEAREST);
                cv::threshold(gt, gt, 0, 255, cv::THRESH_BINARY);
                gtVar = true;
            }
        }
    }
    else {
        sentetikUret(gri, gt);
        gtVar = true;
        cv::imwrite("sentetik.png", gri);
        cv::imwrite("sentetik_gt.png", gt);
    }

    LOG7 = logKernel(7, 1.0);

    cv::Mat sobelX = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat sobelY = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    cv::Mat prewX = (cv::Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
    cv::Mat prewY = (cv::Mat_<float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
    cv::Mat lap = (cv::Mat_<float>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);

    
    std::vector<Yontem> yontemler = {
        {"Sobel", [&](const cv::Mat& g, int t) {
            return esikle(gosterim(buyukluk(gaussla(g), sobelX, sobelY)), t); }},
        {"Prewitt", [&](const cv::Mat& g, int t) {
            return esikle(gosterim(buyukluk(gaussla(g), prewX, prewY)), t); }},
        {"Laplacian", [&](const cv::Mat& g, int t) {
            cv::Mat o; cv::filter2D(gaussla(g), o, CV_32F, lap);
            return esikle(gosterim(o), t); }},
        {"LoG 7x7", [&](const cv::Mat& g, int t) {
            cv::Mat o; cv::filter2D(g, o, CV_32F, LOG7);
            return esikle(gosterim(o), t); }},
        {"Canny", [&](const cv::Mat& g, int t) {
            cv::Mat e; cv::Canny(gaussla(g), e, t, 3 * t);  
            return e; }},
    };

    std::vector<cv::Mat> haritalar;
    std::ofstream csv("sonuclar.csv");
    csv << "yontem,esik,sure_ms,precision,recall,f1\n";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::left << std::setw(12) << "Yontem" << std::setw(8) << "Esik"
        << std::setw(12) << "Sure(ms)" << std::setw(12) << "Precision"
        << std::setw(10) << "Recall" << "F1\n";

    for (auto& y : yontemler) {
        
        int enIyiEsik = 50;
        double enIyiF1 = -1, eP = 0, eR = 0;

        if (gtVar) {
            for (int t = 5; t <= 250; t += 5) {
                double P, R, F1;
                olc(y.kenar(gri, t), gt, 2, P, R, F1);
                if (F1 > enIyiF1) { enIyiF1 = F1; enIyiEsik = t; eP = P; eR = R; }
            }
        }

        double sure = sureOlc(y.kenar, gri, enIyiEsik);
        cv::Mat harita = y.kenar(gri, enIyiEsik);
        haritalar.push_back(harita);

        std::string dosya = "kenar_" + y.ad + ".png";
        for (auto& c : dosya) if (c == ' ') c = '_';
        cv::imwrite(dosya, harita);

        std::cout << std::left << std::setw(12) << y.ad << std::setw(8) << enIyiEsik
            << std::setw(12) << sure;
        if (gtVar)
            std::cout << std::setw(12) << eP << std::setw(10) << eR << enIyiF1 << "\n";
        else
            std::cout << "-\n";

        csv << y.ad << "," << enIyiEsik << "," << sure << ","
            << (gtVar ? eP : 0) << "," << (gtVar ? eR : 0) << "," << (gtVar ? enIyiF1 : 0) << "\n";
    }

    auto kucult = [](const cv::Mat& m) {
        cv::Mat o;
        double oran = 320.0 / m.cols;
        cv::resize(m, o, cv::Size(), oran, oran, cv::INTER_AREA);
        return o;
        };
    cv::Mat bos = cv::Mat::zeros(kucult(gri).size(), CV_8U);
    std::vector<cv::Mat> ust = {
        etiketle(kucult(gri), "Orijinal"),
        gtVar ? etiketle(kucult(gt), "Gercek kenar") : bos,
        etiketle(kucult(haritalar[0]), "Sobel"),
        etiketle(kucult(haritalar[1]), "Prewitt") };
    std::vector<cv::Mat> alt = {
        etiketle(kucult(haritalar[2]), "Laplacian"),
        etiketle(kucult(haritalar[3]), "LoG 7x7"),
        etiketle(kucult(haritalar[4]), "Canny"),
        bos };
    cv::Mat u, a, montaj;
    cv::hconcat(ust, u);
    cv::hconcat(alt, a);
    cv::vconcat(u, a, montaj);
    cv::imwrite("canny_montaj.png", montaj);

    cv::imshow("Karsilastirma", montaj);
    cv::waitKey(0);
    return 0;
}