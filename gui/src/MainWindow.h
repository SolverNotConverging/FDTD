#pragma once
#include <QMainWindow>
#include <QJsonObject>
#include <QProcess>
#include "Project.h"
class Canvas;
class Plot;
class FieldView;
class PortModeView;
class FrequencyFieldView;
class QTreeWidget;
class QTreeWidgetItem;
class QTableWidget;
class QPlainTextEdit;
class QTabWidget;
class QComboBox;
class QLabel;
class QProgressBar;
class QListWidget;
class QDoubleSpinBox;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent=nullptr);
    bool openProject(const QString& path);
    bool openResults(const QString& path);
    bool running() const { return worker_.state()!=QProcess::NotRunning; }
    QString runDirectory() const { return runDirectory_; }
    Project* project() { return &project_; }
    void start(bool preview=false);
    bool smokeImages(const QString& directory);
signals:
    void jobFinished(bool success);
protected:
    void closeEvent(QCloseEvent*) override;
private:
    Project project_; QProcess worker_; QByteArray pending_;
    QJsonObject metadata_,results_; QString projectPath_,runDirectory_,selectedCategory_; int selectedIndex_=-1;
    Canvas* canvas_; FieldView* field_; Plot* sPlot_; Plot* farPlot_; QTabWidget* tabs_;
    PortModeView* modeView_; FrequencyFieldView* monitorView_;
    QTreeWidget* tree_; QTableWidget* properties_; QListWidget* drives_; QDoubleSpinBox* amplitude_;
    QPlainTextEdit* log_; QLabel* summary_; QLabel* coordinates_; QProgressBar* progress_;
    QComboBox* incoming_; QComboBox* sRepresentation_; QComboBox* farRun_; QComboBox* farFrequency_; QComboBox* farRepresentation_; QComboBox* fieldSnapshot_;
    QAction* runAction_; QAction* previewAction_; QAction* cancelAction_;
    bool refreshing_=false,jobCancelled_=false,gotComplete_=false,jobFailed_=false;
    void setupUi(); void refresh(); void select(const QString& category,int index);
    void add(const QString& category,const QString& kind,QJsonObject initial={}); void editSelected(); void deleteSelected();
    void settings(); bool saveProject(bool as=false); bool discardChanges();
    void consume(); void handleEvent(const QJsonObject& event); void cancel();
    void updateS(); void updateFar(); void updateField(); void updateDrives();
    void exportPlot(Plot* plot,bool image); void chooseExample(const QString& name);
    QString python() const; QString sourceRoot() const;
    void appendLog(const QString& text); void loadMetadata(const QJsonObject& data);
};
