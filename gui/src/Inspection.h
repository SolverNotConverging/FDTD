#pragma once
#include <QWidget>
#include <QJsonObject>
class Plot; class FieldView; class QComboBox; class QLabel; class QSlider; class QTimer; class QPushButton;

class PortModeView : public QWidget {
    Q_OBJECT
public:
    explicit PortModeView(QWidget* parent=nullptr);
    void setMetadata(const QJsonObject& metadata);
    void selectPort(int index);
private:
    QJsonObject metadata_;
    QComboBox *port_,*mode_,*frequency_,*component_,*representation_,*dispersion_;
    Plot *field_,*curve_; QLabel* info_;
    void populate(); void updatePlots();
};

class FrequencyFieldView : public QWidget {
    Q_OBJECT
public:
    explicit FrequencyFieldView(QWidget* parent=nullptr);
    void setResults(const QJsonObject& results,const QString& directory);
    void clear(); void render(); void fit();
    bool savePng(const QString& path);
    bool setPhase(int degrees);
private:
    QJsonObject results_; QString directory_;
    QComboBox *run_,*monitor_,*frequency_,*representation_;
    QLabel* info_; QSlider* phase_; QTimer* timer_; QPushButton* play_; FieldView* field_;
    void populateMonitors(); void populateFrequencies(); void updateField();
};
