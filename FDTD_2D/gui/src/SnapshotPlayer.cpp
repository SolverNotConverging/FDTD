#include "SnapshotPlayer.h"
#include <QComboBox>
#include <QPushButton>
#include <QDoubleSpinBox>
#include <QTimer>
#include <QHBoxLayout>
#include <QLabel>
#include <QSignalBlocker>
#include <algorithm>

SnapshotPlayer::SnapshotPlayer(QWidget* parent):QWidget(parent) {
    auto* row=new QHBoxLayout(this); row->setContentsMargins(0,0,0,0);
    selector_=new QComboBox; selector_->setObjectName("timeFieldSnapshot"); selector_->setMinimumWidth(200); selector_->setPlaceholderText("Snapshots available after simulation starts");
    row->addWidget(new QLabel("Snapshot")); row->addWidget(selector_,1);
    play_=new QPushButton("Play"); play_->setObjectName("playTimeFields"); play_->setCheckable(true); row->addWidget(play_);
    auto* rate=new QDoubleSpinBox; rate->setRange(1,30); rate->setDecimals(0); rate->setValue(8); rate->setSuffix(" fps"); rate->setToolTip("Saved-frame playback speed"); row->addWidget(rate);
    timer_=new QTimer(this); timer_->setInterval(125); connect(timer_,&QTimer::timeout,this,[this]{if(isVisible()) advanceFrame();});
    connect(rate,qOverload<double>(&QDoubleSpinBox::valueChanged),this,[this](double fps){timer_->setInterval(qRound(1000/fps));});
    connect(play_,&QPushButton::toggled,this,[this](bool playing){play_->setText(playing?"Pause":"Play"); if(playing) timer_->start(); else timer_->stop();});
    connect(selector_,qOverload<int>(&QComboBox::currentIndexChanged),this,[this]{updatePlayback();}); updatePlayback();
}
QVector<int> SnapshotPlayer::sequence() const {
    QVector<int> indices; const int current=selector_->currentIndex(); if(current<0||current>=frames_.size()) return indices;
    const int run=frames_[current].run;
    for(int i=0;i<frames_.size();++i) if(frames_[i].run==run&&frames_[i].step>=0) indices.append(i);
    std::sort(indices.begin(),indices.end(),[this](int a,int b){return frames_[a].step<frames_[b].step;});
    return indices;
}
void SnapshotPlayer::updatePlayback() {
    const bool enabled=sequence().size()>1;
    if(!enabled) play_->setChecked(false); play_->setEnabled(enabled);
    play_->setToolTip(enabled?"Play saved snapshots chronologically for this run":"Playback requires at least two timed snapshots from the selected run");
}
void SnapshotPlayer::clear() {play_->setChecked(false); frames_.clear(); selector_->clear(); updatePlayback();}
void SnapshotPlayer::setSnapshots(QVector<TimeSnapshot> frames) {
    clear(); frames_=std::move(frames);
    {const QSignalBlocker blocker(selector_); for(const auto& frame:frames_) {selector_->addItem(frame.label,frame.path); selector_->setItemData(selector_->count()-1,frame.title,Qt::UserRole+1);} if(selector_->count()) selector_->setCurrentIndex(0);}
    updatePlayback(); if(selector_->count()) emit selector_->currentIndexChanged(selector_->currentIndex());
}
void SnapshotPlayer::setLiveSnapshot(const TimeSnapshot& frame) {
    // Live files are recycled by the worker; only the latest one is selectable.
    const QSignalBlocker blocker(selector_); clear(); frames_.append(frame); selector_->addItem(frame.label,frame.path); selector_->setItemData(0,frame.title,Qt::UserRole+1); selector_->setCurrentIndex(0); updatePlayback();
}
void SnapshotPlayer::advanceFrame() {
    const auto indices=sequence(); if(indices.size()<2) {play_->setChecked(false);return;}
    const int position=indices.indexOf(selector_->currentIndex()); selector_->setCurrentIndex(indices[(position+1)%indices.size()]);
}
