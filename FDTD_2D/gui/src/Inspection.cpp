#include "Inspection.h"
#include "Plot.h"
#include "FieldView.h"
#include <QComboBox>
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QSlider>
#include <QTimer>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSplitter>
#include <QSignalBlocker>
#include <QJsonArray>
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <complex>
#include <cmath>

namespace {
void exportPlot(QWidget* parent,Plot* plot,bool image) {
    const auto path=QFileDialog::getSaveFileName(parent,image?"Export image":"Export samples",{},image?"PNG (*.png)":"CSV (*.csv)");
    if(!path.isEmpty()&&!(image?plot->savePng(path):plot->saveCsv(path))) QMessageBox::warning(parent,"Export failed","Could not write the selected file.");
}
}
PortModeView::PortModeView(QWidget* parent):QWidget(parent) {
    auto* layout=new QVBoxLayout(this); auto* controls=new QHBoxLayout;
    port_=new QComboBox; port_->setObjectName("modePort"); mode_=new QComboBox; mode_->setObjectName("modeIndex"); frequency_=new QComboBox; frequency_->setObjectName("modeFrequency");
    controls->addWidget(new QLabel("Port")); controls->addWidget(port_,1); controls->addWidget(mode_); controls->addWidget(new QLabel("Anchor")); controls->addWidget(frequency_); layout->addLayout(controls);
    info_=new QLabel; info_->setWordWrap(true); layout->addWidget(info_);
    auto* fields=new QHBoxLayout; component_=new QComboBox; representation_=new QComboBox; representation_->addItems({"Magnitude","Real","Imaginary","Phase (deg)"});
    fields->addWidget(component_); fields->addWidget(representation_); fields->addStretch(); auto* fieldCsv=new QPushButton("Field CSV"); auto* fieldPng=new QPushButton("Field PNG"); fields->addWidget(fieldCsv); fields->addWidget(fieldPng); layout->addLayout(fields);
    field_=new Plot; field_->setObjectName("modeFieldPlot"); layout->addWidget(field_,1);
    auto* dispersions=new QHBoxLayout; dispersion_=new QComboBox; dispersion_->addItems({"Phase constant beta","Attenuation alpha","Effective index beta/k0","Tracking overlap"}); dispersions->addWidget(new QLabel("Dispersion")); dispersions->addWidget(dispersion_,1);
    auto* curveCsv=new QPushButton("Dispersion CSV"); auto* curvePng=new QPushButton("Dispersion PNG"); dispersions->addWidget(curveCsv); dispersions->addWidget(curvePng); layout->addLayout(dispersions);
    curve_=new Plot; curve_->setObjectName("modeDispersionPlot"); layout->addWidget(curve_,1);
    connect(port_,qOverload<int>(&QComboBox::currentIndexChanged),this,[this]{populate();});
    for(auto* combo:{mode_,frequency_,component_,representation_,dispersion_}) connect(combo,qOverload<int>(&QComboBox::currentIndexChanged),this,[this]{updatePlots();});
    connect(fieldCsv,&QPushButton::clicked,this,[this]{exportPlot(this,field_,false);}); connect(fieldPng,&QPushButton::clicked,this,[this]{exportPlot(this,field_,true);});
    connect(curveCsv,&QPushButton::clicked,this,[this]{exportPlot(this,curve_,false);}); connect(curvePng,&QPushButton::clicked,this,[this]{exportPlot(this,curve_,true);});
    setMetadata({});
}
void PortModeView::setMetadata(const QJsonObject& metadata) {
    metadata_=metadata; {const QSignalBlocker block(port_); port_->clear(); for(const auto p:metadata["port_modes"].toArray()) port_->addItem(p.toObject()["name"].toString());} populate();
}
void PortModeView::selectPort(int index) { port_->setCurrentIndex(index); }
void PortModeView::populate() {
    const QSignalBlocker a(mode_),b(frequency_),c(component_); mode_->clear(); frequency_->clear(); component_->clear();
    const auto ports=metadata_["port_modes"].toArray(); if(port_->currentIndex()<0) { updatePlots(); return; }
    const auto p=ports[port_->currentIndex()].toObject();
    for(const auto m:p["modes"].toArray()) {const auto mode=m.toObject(); const auto family=mode["family"].toString(); mode_->addItem("Mode "+QString::number(mode["index"].toInt())+(family.isEmpty()?QString():" · "+family));}
    for(const auto f:p["frequencies_ghz"].toArray()) frequency_->addItem(QString::number(f.toDouble(),'g',8)+" GHz"); frequency_->setCurrentIndex(frequency_->count()/2);
    component_->addItem(p["scalar_label"].toString(),"q"); component_->addItem(p["tangent_label"].toString(),"p"); updatePlots();
}
void PortModeView::updatePlots() {
    if(port_->currentIndex()<0||mode_->currentIndex()<0||frequency_->currentIndex()<0) { info_->setText("Generate the mesh to solve and track waveguide modes at every frequency anchor."); field_->setMessage("No tracked waveguide modes. Older results need a new mesh preview."); curve_->setMessage("Dispersion appears after mesh generation."); return; }
    const auto p=metadata_["port_modes"].toArray()[port_->currentIndex()].toObject(),m=p["modes"].toArray()[mode_->currentIndex()].toObject();
    const int f=frequency_->currentIndex(),representation=representation_->currentIndex(); const bool valid=m["valid"].toArray()[f].toBool();
    info_->setText(QString("%1 · Re(beta) %2, Im(beta) %3 rad/m · overlap %4 · %5\nexp(+i omega t - i beta s); alpha = -Im(beta). Tangential field uses the outgoing-wave basis.")
        .arg(valid?"Propagating: 1 W power normalization":"Evanescent: eigenvector scale, no power normalization")
        .arg(m["beta_real_rad_m"].toArray()[f].toDouble(),0,'g',7).arg(m["beta_imag_rad_m"].toArray()[f].toDouble(),0,'g',7).arg(m["overlap"].toArray()[f].toDouble(),0,'g',5).arg(frequency_->currentText()));
    const auto prefix=component_->currentData().toString(); const auto xs=p["transverse_mm"].toArray();
    const auto real=m[prefix+"_real"].toArray()[f].toArray(),imag=m[prefix+"_imag"].toArray()[f].toArray();
    Curve field{p["name"].toString()+":"+QString::number(mode_->currentIndex())+" · "+frequency_->currentText(),{},QColor("#147eb3")};
    for(int i=0;i<xs.size();++i) { const std::complex<double> z(real[i].toDouble(),imag[i].toDouble()); const double value=representation==0?std::abs(z):representation==1?z.real():representation==2?z.imag():std::arg(z)*180./3.141592653589793; field.points.append({xs[i].toDouble(),value}); }
    const auto unit=p[prefix=="q"?"scalar_unit":"tangent_unit"].toString();
    field_->setCurves({field},"Transverse "+QString(p["axis"].toString()=="x"?"Y":"X")+" (mm)",representation==3?"Phase (deg)":component_->currentText()+" ("+unit+(valid?" / sqrt(W))":", eigenvector scale)"));
    const QStringList keys{"beta_real_rad_m","attenuation_np_m","effective_index","overlap"},units{"beta (rad/m)","alpha (Np/m)","beta/k0","Tracking overlap"};
    QVector<Curve> curves; const auto frequencies=p["frequencies_ghz"].toArray(); int k=0;
    for(const auto value:p["modes"].toArray()) { const auto mode=value.toObject(); Curve c{"Mode "+QString::number(mode["index"].toInt()),{},k++%2?QColor("#d35c57"):QColor("#147eb3")};
        const auto ys=mode[keys[dispersion_->currentIndex()]].toArray(); for(int i=0;i<frequencies.size();++i) c.points.append({frequencies[i].toDouble(),ys[i].toDouble()}); curves.append(c); }
    curve_->setCurves(curves,"Anchor frequency (GHz)",units[dispersion_->currentIndex()]);
}
FrequencyFieldView::FrequencyFieldView(QWidget* parent):QWidget(parent) {
    auto* layout=new QVBoxLayout(this); auto* row=new QHBoxLayout; run_=new QComboBox; monitor_=new QComboBox; frequency_=new QComboBox; representation_=new QComboBox;
    run_->setObjectName("monitorRun"); monitor_->setObjectName("fieldMonitor"); frequency_->setObjectName("monitorFrequency"); representation_->setObjectName("monitorRepresentation");
    representation_->addItems({"Magnitude","Real","Imaginary","Phase (deg)","Harmonic animation"});
    row->addWidget(run_); row->addWidget(monitor_,1); row->addWidget(frequency_); row->addWidget(representation_); layout->addLayout(row);
    auto* animation=new QHBoxLayout; play_=new QPushButton("Play"); play_->setCheckable(true); play_->setObjectName("playMonitor"); phase_=new QSlider(Qt::Horizontal); phase_->setObjectName("monitorPhase"); phase_->setRange(0,359);
    animation->addWidget(play_); animation->addWidget(new QLabel("Phase")); animation->addWidget(phase_,1); auto* edges=new QCheckBox("Cell edges"); animation->addWidget(edges); auto* fitButton=new QPushButton("Fit"); animation->addWidget(fitButton); auto* png=new QPushButton("PNG"); animation->addWidget(png); layout->addLayout(animation);
    info_=new QLabel("Add a rectangular field monitor, then run the simulation."); info_->setWordWrap(true); layout->addWidget(info_); field_=new FieldView; layout->addWidget(field_,1);
    timer_=new QTimer(this); timer_->setInterval(33); connect(timer_,&QTimer::timeout,this,[this]{if(isVisible()) phase_->setValue((phase_->value()+4)%360);});
    connect(play_,&QPushButton::toggled,this,[this](bool playing){if(playing) { representation_->setCurrentIndex(4); timer_->start(); } else timer_->stop(); play_->setText(playing?"Pause":"Play");});
    connect(phase_,&QSlider::valueChanged,this,[this](int degrees){field_->setHarmonicPhase(degrees);});
    connect(run_,qOverload<int>(&QComboBox::currentIndexChanged),this,[this]{populateMonitors();}); connect(monitor_,qOverload<int>(&QComboBox::currentIndexChanged),this,[this]{populateFrequencies();});
    connect(frequency_,qOverload<int>(&QComboBox::currentIndexChanged),this,[this]{updateField();}); connect(representation_,qOverload<int>(&QComboBox::currentIndexChanged),this,[this]{if(representation_->currentIndex()!=4) play_->setChecked(false); updateField();});
    connect(edges,&QCheckBox::toggled,field_,&FieldView::setEdges); connect(fitButton,&QPushButton::clicked,field_,&FieldView::fit);
    connect(png,&QPushButton::clicked,this,[this]{const auto path=QFileDialog::getSaveFileName(this,"Export monitored field",{},"PNG (*.png)"); if(!path.isEmpty()&&!savePng(path)) QMessageBox::warning(this,"Export failed","Could not write the image.");});
    clear();
}
void FrequencyFieldView::clear() { timer_->stop(); play_->setChecked(false); results_={}; run_->clear(); monitor_->clear(); frequency_->clear(); field_->clear(); info_->setText("Add a rectangular field monitor, then run the simulation. Displays Ez (TM) or Hz (TE)."); play_->setEnabled(false); phase_->setEnabled(false); }
void FrequencyFieldView::setResults(const QJsonObject& results,const QString& directory) {
    clear(); results_=results; directory_=directory; {const QSignalBlocker block(run_); const auto runs=results["runs"].toArray(); for(int i=0;i<runs.size();++i) run_->addItem("Run "+QString::number(i+1));} populateMonitors();
}
void FrequencyFieldView::populateMonitors() {
    {const QSignalBlocker block(monitor_); monitor_->clear(); if(run_->currentIndex()>=0) for(const auto m:results_["runs"].toArray()[run_->currentIndex()].toObject()["monitors"].toArray()) monitor_->addItem(m.toObject()["name"].toString());} populateFrequencies();
}
void FrequencyFieldView::populateFrequencies() {
    field_->clear();
    {const QSignalBlocker block(frequency_); frequency_->clear(); if(monitor_->currentIndex()>=0) { const auto m=results_["runs"].toArray()[run_->currentIndex()].toObject()["monitors"].toArray()[monitor_->currentIndex()].toObject(); for(const auto f:m["frequencies_ghz"].toArray()) frequency_->addItem(QString::number(f.toDouble(),'g',8)+" GHz"); frequency_->setCurrentIndex(frequency_->count()/2); }} updateField();
}
void FrequencyFieldView::updateField() {
    if(run_->currentIndex()<0||monitor_->currentIndex()<0||frequency_->currentIndex()<0) { field_->clear(); info_->setText("No frequency monitor results. Add a monitor before running."); play_->setEnabled(false); phase_->setEnabled(false); return; }
    const auto m=results_["runs"].toArray()[run_->currentIndex()].toObject()["monitors"].toArray()[monitor_->currentIndex()].toObject();
    QString path=m["field_files"].toArray()[frequency_->currentIndex()].toString(); if(!QFileInfo::exists(path)) path=directory_+"/"+QFileInfo(path).fileName();
    const int representation=representation_->currentIndex(); const QStringList arrays{"magnitude","real","imaginary","phase","animated"};
    const QString title=m["field_label"].toString()+" · "+frequency_->currentText()+" ("+(representation==3?QString("deg"):m["unit"].toString())+")";
    const bool loaded=field_->load(path,title,representation!=0,arrays[representation]); play_->setEnabled(loaded); phase_->setEnabled(loaded&&representation==4);
    info_->setText(loaded?"Regional raw DFT: sum(field × exp(-i omega t) × dt). Animation shows Re(DFT × exp(+i phase)); 0.37 displayed cycles/s. Physical period: "+QString::number(1000./m["frequencies_ghz"].toArray()[frequency_->currentIndex()].toDouble(),'g',6)+" ps.":"Monitor field file is missing or invalid.");
    if(loaded&&representation==4) field_->setHarmonicPhase(phase_->value()); if(!loaded) { timer_->stop(); play_->setChecked(false); field_->clear(); }
}
void FrequencyFieldView::render() { field_->render(); }
void FrequencyFieldView::fit() { field_->fit(); }
bool FrequencyFieldView::savePng(const QString& path) { return field_->savePng(path); }
bool FrequencyFieldView::setPhase(int degrees) { representation_->setCurrentIndex(4); phase_->setValue(degrees); return field_->setHarmonicPhase(degrees); }
