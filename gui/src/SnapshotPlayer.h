#pragma once
#include <QWidget>
#include <QVector>
class QComboBox;
class QPushButton;
class QTimer;

struct TimeSnapshot {
    QString label,path,title;
    int run=0,step=-1;
    double timeNs=0;
};
class SnapshotPlayer : public QWidget {
    Q_OBJECT
public:
    explicit SnapshotPlayer(QWidget* parent=nullptr);
    QComboBox* selector() const {return selector_;}
    void setSnapshots(QVector<TimeSnapshot> frames);
    void setLiveSnapshot(const TimeSnapshot& frame);
    void clear();
    void advanceFrame();
private:
    QVector<TimeSnapshot> frames_;
    QComboBox* selector_;
    QPushButton* play_;
    QTimer* timer_;
    QVector<int> sequence() const;
    void updatePlayback();
};
