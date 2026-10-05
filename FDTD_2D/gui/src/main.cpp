#include "MainWindow.h"
#include <QVTKOpenGLNativeWidget.h>
#include <QApplication>
#include <QSurfaceFormat>
#include <QCommandLineParser>
#include <QTimer>
#include <QFile>
#include <QJsonDocument>
#include <QDir>
#include <QAction>

int main(int argc,char** argv) {
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());
    QApplication app(argc,argv); QApplication::setApplicationName("FDTD Studio"); QApplication::setOrganizationName("FDTD");
    app.setStyle("Fusion"); app.setFont(QFont("Segoe UI",10));
    app.setStyleSheet("QMainWindow,QDialog{background:#f1f5f9;} QToolBar{spacing:4px;padding:6px;background:#e8eff5;border-bottom:1px solid #ccd8e3;} QDockWidget::title{padding:7px;background:#e4edf5;color:#344b62;} QTabBar::tab{padding:10px 20px;} QTabBar::tab:selected{background:white;color:#137da8;} QPushButton{padding:6px 10px;} QProgressBar{border:1px solid #c7d5e1;border-radius:3px;text-align:center;} QProgressBar::chunk{background:#2699a6;} QTreeWidget,QTableWidget,QListWidget,QPlainTextEdit{background:white;alternate-background-color:#f5f8fb;} ");
    QCommandLineParser parser; parser.setApplicationDescription("Native modeller for the general 2D FDTD solver"); parser.addHelpOption();
    parser.addOption({"project","Open a project","path"}); parser.addOption({"results","Open simulation results","path"});
    parser.addOption({"run","Run the project after opening"}); parser.addOption({"preview","Generate mesh after opening"});
    parser.addOption({"smoke-test","Capture all native GUI tabs then exit","directory"}); parser.addOption({"exit-after-run","Exit after the worker finishes (integration testing)"});
    parser.addOption({"cancel-after-ms","Request cancellation after a delay (integration testing)","milliseconds"}); parser.process(app);
    MainWindow window;
    if(parser.isSet("project")) { if(!window.openProject(parser.value("project"))) return 2; }
    else if(!parser.isSet("results")) window.openProject(QString::fromUtf8(FDTD_SOURCE_ROOT)+"/FDTD_2D/gui/examples/cylinder.fdtd.json");
    if(parser.isSet("results")&&!window.openResults(parser.value("results"))) return 3;
    window.show();
    if(parser.isSet("exit-after-run")) QObject::connect(&window,&MainWindow::jobFinished,&app,[&](bool success){if(parser.isSet("smoke-test")&&success) success=window.smokeImages(parser.value("smoke-test")); app.exit(success?0:4);});
    if(parser.isSet("run")||parser.isSet("preview")) QTimer::singleShot(200,&window,[&]{window.start(parser.isSet("preview"));});
    else if(parser.isSet("smoke-test")) QTimer::singleShot(1200,&window,[&]{app.exit(window.smokeImages(parser.value("smoke-test"))?0:5);});
    if(parser.isSet("cancel-after-ms")) QTimer::singleShot(parser.value("cancel-after-ms").toInt(),&window,[&]{if(auto* action=window.findChild<QAction*>("cancelSimulation")) action->trigger();});
    return app.exec();
}
