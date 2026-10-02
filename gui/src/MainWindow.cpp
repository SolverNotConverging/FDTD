#include "MainWindow.h"
#include "Canvas.h"
#include "Plot.h"
#include "FieldView.h"
#include "Inspection.h"
#include "Dialogs.h"
#include <QApplication>
#include <QAction>
#include <QActionGroup>
#include <QMenuBar>
#include <QToolBar>
#include <QDockWidget>
#include <QTreeWidget>
#include <QTableWidget>
#include <QHeaderView>
#include <QPlainTextEdit>
#include <QTabWidget>
#include <QComboBox>
#include <QListWidget>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QProgressBar>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QCheckBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QSaveFile>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QDateTime>
#include <QSettings>
#include <QStandardPaths>
#include <QDesktopServices>
#include <QUrl>
#include <QUuid>
#include <QTimer>
#include <QEventLoop>
#include <QSlider>
#include <QCloseEvent>
#include <QSignalBlocker>
#include <QPainter>
#include <QDir>
#include <QLineF>
#include <complex>
#include <limits>
#include <cmath>
#include <algorithm>

namespace {
double numeric(const QJsonValue& value) { return value.isDouble()?value.toDouble():std::numeric_limits<double>::quiet_NaN(); }
QString channel(const QJsonValue& value) { const auto pair=value.toArray(); return pair[0].toString()+":"+QString::number(pair[1].toInt()); }
const QVector<QColor> curveColors={QColor("#147eb3"),QColor("#d35c57"),QColor("#33a17e"),QColor("#9973c5"),QColor("#dd923c"),QColor("#4e667e")};
QJsonObject readJson(const QString& path,QString* error) {
    QFile file(path); if(!file.open(QIODevice::ReadOnly)) { *error=file.errorString(); return {}; }
    QJsonParseError parse; const auto document=QJsonDocument::fromJson(file.readAll(),&parse);
    if(parse.error!=QJsonParseError::NoError||!document.isObject()) { *error="Invalid JSON: "+parse.errorString(); return {}; }
    return document.object();
}
}
MainWindow::MainWindow(QWidget* parent):QMainWindow(parent),project_(this),worker_(this) {
    setObjectName("FDTDStudio"); resize(1450,940); setupUi();
    connect(&project_,&Project::changed,this,[this]{
        metadata_={}; results_={}; field_->clear(); canvas_->setCompiled({});
        modeView_->setMetadata({}); monitorView_->clear();
        sPlot_->setMessage("Run an S-parameter study to inspect the complete matrix."); farPlot_->setMessage("Run a simulation to inspect its closed-contour far field.");
        fieldSnapshot_->clear(); refresh();
    });
    connect(project_.undoStack(),&QUndoStack::cleanChanged,this,[this]{setWindowModified(project_.dirty());});
    connect(&worker_,&QProcess::readyReadStandardOutput,this,&MainWindow::consume);
    connect(&worker_,&QProcess::readyReadStandardError,this,[this]{appendLog(QString::fromUtf8(worker_.readAllStandardError()));});
    connect(&worker_,&QProcess::errorOccurred,this,[this](QProcess::ProcessError error){
        if(error==QProcess::FailedToStart) { jobFailed_=true; appendLog("Could not start Python. Set the interpreter in Simulation > Python runtime."); runAction_->setEnabled(true); previewAction_->setEnabled(true); cancelAction_->setEnabled(false); canvas_->setEnabled(true); drives_->setEnabled(true); amplitude_->setEnabled(true); emit jobFinished(false); }
    });
    connect(&worker_,qOverload<int,QProcess::ExitStatus>(&QProcess::finished),this,[this](int code,QProcess::ExitStatus status){
        consume(); runAction_->setEnabled(true); previewAction_->setEnabled(true); cancelAction_->setEnabled(false);
        canvas_->setEnabled(true); drives_->setEnabled(true); amplitude_->setEnabled(true);
        const bool ok=code==0&&status==QProcess::NormalExit&&gotComplete_&&!jobFailed_&&!jobCancelled_;
        statusBar()->showMessage(jobCancelled_?"Simulation cancelled":ok?"Completed":"Simulation failed — see the log",15000);
        if(ok) progress_->setValue(1000); else progress_->setValue(0);
        emit jobFinished(ok);
    });
    refresh();
    restoreGeometry(QSettings().value("geometry").toByteArray());
    restoreState(QSettings().value("layout").toByteArray());
}
void MainWindow::setupUi() {
    auto* file=menuBar()->addMenu("&File");
    auto* fresh=file->addAction("New project"); fresh->setShortcut(QKeySequence::New); connect(fresh,&QAction::triggered,this,[this]{if(running()||!discardChanges()) return; projectPath_.clear(); project_.reset(Project::empty()); canvas_->fitModel();});
    auto* open=file->addAction("Open project…"); open->setShortcut(QKeySequence::Open); connect(open,&QAction::triggered,this,[this]{if(running()||!discardChanges()) return; const auto path=QFileDialog::getOpenFileName(this,"Open FDTD project",{},"FDTD projects (*.fdtd.json *.json)"); if(!path.isEmpty()) openProject(path);});
    auto* save=file->addAction("Save project"); save->setShortcut(QKeySequence::Save); connect(save,&QAction::triggered,this,[this]{saveProject();});
    auto* saveAs=file->addAction("Save project as…"); saveAs->setShortcut(QKeySequence::SaveAs); connect(saveAs,&QAction::triggered,this,[this]{saveProject(true);});
    file->addSeparator(); auto* openResult=file->addAction("Open simulation results…"); connect(openResult,&QAction::triggered,this,[this]{if(running()||!discardChanges()) return; const auto path=QFileDialog::getOpenFileName(this,"Open results",{},"FDTD results (results.json)"); if(!path.isEmpty()) openResults(path);});
    auto* folder=file->addAction("Open run folder"); connect(folder,&QAction::triggered,this,[this]{if(!runDirectory_.isEmpty()) QDesktopServices::openUrl(QUrl::fromLocalFile(runDirectory_));});
    file->addSeparator(); auto* quit=file->addAction("Exit"); connect(quit,&QAction::triggered,this,&QWidget::close);
    auto* edit=menuBar()->addMenu("&Edit"); edit->addAction(project_.undoStack()->createUndoAction(this)); edit->actions().last()->setShortcut(QKeySequence::Undo); edit->addAction(project_.undoStack()->createRedoAction(this)); edit->actions().last()->setShortcut(QKeySequence::Redo);
    auto* property=edit->addAction("Edit selected item…"); connect(property,&QAction::triggered,this,&MainWindow::editSelected);
    auto* remove=edit->addAction("Delete selected item"); remove->setShortcut(QKeySequence::Delete); connect(remove,&QAction::triggered,this,&MainWindow::deleteSelected);
    auto* geometry=menuBar()->addMenu("&Model");
    for(const auto& kind:{QString("rectangle"),QString("circle"),QString("polygon"),QString("sheet")}) { auto* action=geometry->addAction("Add "+kind+"…"); connect(action,&QAction::triggered,this,[this,kind]{add("objects",kind);}); }
    geometry->addSeparator();
    for(const auto& kind:{QString("lumped"),QString("waveguide")}) { auto* action=geometry->addAction("Add "+kind+" port…"); connect(action,&QAction::triggered,this,[this,kind]{add("ports",kind);}); }
    auto* plane=geometry->addAction("Add plane wave…"); connect(plane,&QAction::triggered,this,[this]{add("sources","plane");});
    auto* monitor=geometry->addAction("Add field monitor…"); connect(monitor,&QAction::triggered,this,[this]{add("monitors","monitor");});
    auto* simulation=menuBar()->addMenu("&Simulation"); auto* configure=simulation->addAction("Settings…"); connect(configure,&QAction::triggered,this,&MainWindow::settings);
    previewAction_=simulation->addAction("Generate mesh"); previewAction_->setShortcut(Qt::Key_F6); connect(previewAction_,&QAction::triggered,this,[this]{start(true);});
    runAction_=simulation->addAction("Run simulation"); runAction_->setShortcut(Qt::Key_F5); connect(runAction_,&QAction::triggered,this,[this]{start();});
    cancelAction_=simulation->addAction("Stop"); cancelAction_->setObjectName("cancelSimulation"); cancelAction_->setEnabled(false); connect(cancelAction_,&QAction::triggered,this,&MainWindow::cancel);
    auto* runtime=simulation->addAction("Python runtime…"); connect(runtime,&QAction::triggered,this,[this]{const auto path=QFileDialog::getOpenFileName(this,"Choose Python interpreter",python(),"Python executable (python.exe python3 python)"); if(!path.isEmpty()) { QSettings().setValue("python",path); appendLog("Python: "+path); }});
    auto* examples=menuBar()->addMenu("&Examples");
    for(const auto& pair:{qMakePair(QString("PEC cylinder / plane wave"),QString("cylinder")),qMakePair(QString("Matched two-port guide"),QString("waveguide")),qMakePair(QString("Thin-film mixed geometry"),QString("thin-film")),qMakePair(QString("Periodic leaky-wave antenna"),QString("leaky-wave"))}) { auto* action=examples->addAction(pair.first); connect(action,&QAction::triggered,this,[this,pair]{chooseExample(pair.second);}); }
    auto* help=menuBar()->addMenu("&Help"); auto* doc=help->addAction("Documentation"); connect(doc,&QAction::triggered,this,[this]{QDesktopServices::openUrl(QUrl::fromLocalFile(sourceRoot()+"/gui/README.md"));});
    auto* about=help->addAction("About FDTD Studio"); connect(about,&QAction::triggered,this,[this]{QMessageBox::about(this,"FDTD Studio","Native C++ / Qt / VTK modelling and result inspection.\nGeometry-first conformal TE/TM reference solver.\nCoordinates: mm · Frequency: GHz · Time: ns\nQt and VTK are dynamically linked; see gui/THIRD_PARTY.md.");});
    auto* toolbar=addToolBar("Model tools"); toolbar->setObjectName("modelToolbar"); toolbar->setMovable(false); toolbar->addAction(configure); toolbar->addAction(previewAction_); toolbar->addAction(runAction_); toolbar->addAction(cancelAction_); toolbar->addSeparator();
    auto* tools=new QActionGroup(this); tools->setExclusive(true);
    for(const auto& pair:{qMakePair(QString("Select / move"),QString("select")),qMakePair(QString("Rectangle"),QString("rectangle")),qMakePair(QString("Circle"),QString("circle")),qMakePair(QString("Polygon"),QString("polygon")),qMakePair(QString("Sheet"),QString("sheet")),qMakePair(QString("Lumped"),QString("lumped")),qMakePair(QString("Guide port"),QString("waveguide")),qMakePair(QString("Monitor"),QString("monitor"))}) {
        auto* action=toolbar->addAction(pair.first); action->setCheckable(true); action->setData(pair.second); tools->addAction(action); if(pair.second=="select") action->setChecked(true);
        connect(action,&QAction::triggered,this,[this,pair]{canvas_->setTool(pair.second); tabs_->setCurrentIndex(0); statusBar()->showMessage(pair.second=="polygon"?"Click vertices; right-click to finish. Esc cancels.":"Draw in the model view. Wheel zooms; select mode pans and moves items.",12000);});
    }
    auto* fit=toolbar->addAction("Fit"); connect(fit,&QAction::triggered,this,[this]{canvas_->fitModel(); field_->fit();});
    auto* snap=new QDoubleSpinBox; snap->setRange(0,100); snap->setDecimals(3); snap->setValue(.1); snap->setSuffix(" mm snap"); toolbar->addWidget(snap);
    tabs_=new QTabWidget; canvas_=new Canvas; tabs_->addTab(canvas_,"Model"); setCentralWidget(tabs_);
    connect(snap,qOverload<double>(&QDoubleSpinBox::valueChanged),canvas_,&Canvas::setSnap);
    connect(canvas_,&Canvas::selected,this,&MainWindow::select); connect(canvas_,&Canvas::moved,&project_,&Project::move);
    connect(canvas_,&Canvas::coordinate,this,[this](double x,double y){coordinates_->setText(QString("X %1   Y %2 mm").arg(x,0,'f',3).arg(y,0,'f',3));});
    connect(canvas_,&Canvas::drawn,this,[this,tools](const QString& kind,const QVector<QPointF>& points){
        tools->actions().first()->setChecked(true); if(running()) return;
        QJsonObject object; auto point=[](const QPointF& p){return QJsonArray{p.x(),p.y()};};
        if(kind=="rectangle"||kind=="monitor") { object["x"]=std::min(points[0].x(),points[1].x()); object["y"]=std::min(points[0].y(),points[1].y()); object["width"]=std::abs(points[1].x()-points[0].x()); object["height"]=std::abs(points[1].y()-points[0].y()); }
        else if(kind=="circle") { object["x"]=points[0].x(); object["y"]=points[0].y(); object["radius"]=QLineF(points[0],points[1]).length(); }
        else if(kind=="polygon") { QJsonArray vertices; for(const auto& p:points) vertices.append(point(p)); object["vertices"]=vertices; }
        else if(kind=="sheet"||kind=="lumped") { object["start"]=point(points[0]); if(points.size()>1) object["end"]=point(points[1]); }
        else if(kind=="waveguide") { const bool x=std::abs(points[1].y()-points[0].y())>std::abs(points[1].x()-points[0].x()); object["axis"]=x?"x":"y"; object["position"]=x?points[0].x():points[0].y(); const double a=x?points[0].y():points[0].x(),b=x?points[1].y():points[1].x(); object["span"]=QJsonArray{std::min(a,b),std::max(a,b)}; }
        add(kind=="monitor"?"monitors":kind=="lumped"||kind=="waveguide"?"ports":"objects",kind,object);
    });
    auto* fieldPage=new QWidget; auto* fl=new QVBoxLayout(fieldPage); auto* fieldTools=new QHBoxLayout;
    fieldSnapshot_=new QComboBox; fieldSnapshot_->setMinimumWidth(200); fieldTools->addWidget(new QLabel("Snapshot")); fieldTools->addWidget(fieldSnapshot_,1);
    auto* edges=new QCheckBox("Cell edges"); fieldTools->addWidget(edges); auto* fieldImage=new QPushButton("Export PNG"); fieldTools->addWidget(fieldImage); fl->addLayout(fieldTools);
    field_=new FieldView; fl->addWidget(field_,1); tabs_->addTab(fieldPage,"Mesh & fields"); connect(edges,&QCheckBox::toggled,field_,&FieldView::setEdges); connect(fieldSnapshot_,qOverload<int>(&QComboBox::currentIndexChanged),this,[this]{updateField();});
    connect(fieldImage,&QPushButton::clicked,this,[this]{const auto path=QFileDialog::getSaveFileName(this,"Export field image",{},"PNG image (*.png)"); if(!path.isEmpty()&&!field_->savePng(path)) QMessageBox::warning(this,"Export failed","Could not write the image.");});
    auto* sPage=new QWidget; auto* sl=new QVBoxLayout(sPage); auto* sTools=new QHBoxLayout; incoming_=new QComboBox; sRepresentation_=new QComboBox; sRepresentation_->addItems({"Magnitude (dB)","Phase (deg)","Real","Imaginary"}); sTools->addWidget(new QLabel("Incident channel")); sTools->addWidget(incoming_,1); sTools->addWidget(sRepresentation_);
    sPlot_=new Plot; auto* sCsv=new QPushButton("CSV"); auto* sPng=new QPushButton("PNG"); sTools->addWidget(sCsv); sTools->addWidget(sPng); sl->addLayout(sTools); sl->addWidget(sPlot_,1); tabs_->addTab(sPage,"S parameters");
    connect(incoming_,qOverload<int>(&QComboBox::currentIndexChanged),this,[this]{updateS();}); connect(sRepresentation_,qOverload<int>(&QComboBox::currentIndexChanged),this,[this]{updateS();}); connect(sCsv,&QPushButton::clicked,this,[this]{exportPlot(sPlot_,false);}); connect(sPng,&QPushButton::clicked,this,[this]{exportPlot(sPlot_,true);});
    auto* farPage=new QWidget; auto* ffl=new QVBoxLayout(farPage); auto* farTools=new QHBoxLayout; farRun_=new QComboBox; farFrequency_=new QComboBox; farRepresentation_=new QComboBox; farRepresentation_->addItems({"Relative power (dB)","Power / incident W","Raw DFT power","Scalar phase (deg)"}); farTools->addWidget(farRun_,1); farTools->addWidget(farFrequency_); farTools->addWidget(farRepresentation_);
    farPlot_=new Plot; auto* farCsv=new QPushButton("CSV"); auto* farPng=new QPushButton("PNG"); farTools->addWidget(farCsv); farTools->addWidget(farPng); ffl->addLayout(farTools); ffl->addWidget(farPlot_,1); tabs_->addTab(farPage,"Far field");
    for(auto* combo:{farRun_,farFrequency_,farRepresentation_}) connect(combo,qOverload<int>(&QComboBox::currentIndexChanged),this,[this]{updateFar();}); connect(farCsv,&QPushButton::clicked,this,[this]{exportPlot(farPlot_,false);}); connect(farPng,&QPushButton::clicked,this,[this]{exportPlot(farPlot_,true);});
    modeView_=new PortModeView; tabs_->addTab(modeView_,"Port modes");
    monitorView_=new FrequencyFieldView; tabs_->addTab(monitorView_,"Frequency fields");
    auto* showMesh=new QCheckBox("Show mesh"); showMesh->setObjectName("showSimulationMesh"); showMesh->setChecked(true); toolbar->addWidget(showMesh);
    connect(showMesh,&QCheckBox::toggled,canvas_,&Canvas::setShowMesh);
    edges->setChecked(true);
    auto* modelDock=new QDockWidget("Model tree",this); tree_=new QTreeWidget; tree_->setHeaderLabels({"Item","Material / kind","Rank"}); tree_->setColumnWidth(0,130); modelDock->setWidget(tree_); addDockWidget(Qt::LeftDockWidgetArea,modelDock); modelDock->setMinimumWidth(285);
    connect(tree_,&QTreeWidget::itemSelectionChanged,this,[this]{if(refreshing_) return; const auto selected=tree_->selectedItems(); if(selected.isEmpty()) return; auto* item=selected.first(); select(item->data(0,Qt::UserRole).toString(),item->data(0,Qt::UserRole+1).toInt());}); connect(tree_,&QTreeWidget::itemDoubleClicked,this,[this]{editSelected();});
    auto* propDock=new QDockWidget("Properties",this); auto* propPage=new QWidget; auto* pl=new QVBoxLayout(propPage); properties_=new QTableWidget(0,2); properties_->setHorizontalHeaderLabels({"Property","Value"}); properties_->horizontalHeader()->setStretchLastSection(true); properties_->verticalHeader()->hide(); properties_->setEditTriggers(QAbstractItemView::NoEditTriggers); pl->addWidget(properties_); auto* editButton=new QPushButton("Edit selected…"); pl->addWidget(editButton); connect(editButton,&QPushButton::clicked,this,&MainWindow::editSelected); propDock->setWidget(propPage); addDockWidget(Qt::RightDockWidgetArea,propDock); propDock->setMinimumWidth(280);
    auto* driveDock=new QDockWidget("Excitation / receiving",this); auto* drivePage=new QWidget; auto* dl=new QVBoxLayout(drivePage); drives_=new QListWidget; dl->addWidget(new QLabel("Checked channels transmit in a coherent run.\nAll ports receive; S studies drive all channels.")); dl->addWidget(drives_); auto* amplitudeRow=new QHBoxLayout; amplitude_=new QDoubleSpinBox; amplitude_->setRange(-1e9,1e9); amplitude_->setDecimals(6); amplitude_->setValue(1); amplitudeRow->addWidget(new QLabel("Drive amplitude")); amplitudeRow->addWidget(amplitude_); dl->addLayout(amplitudeRow); driveDock->setWidget(drivePage); addDockWidget(Qt::RightDockWidgetArea,driveDock);
    connect(drives_,&QListWidget::itemChanged,this,[this]{updateDrives();});
    connect(drives_,&QListWidget::currentRowChanged,this,[this](int row){if(row<0) return; const QSignalBlocker blocker(amplitude_); amplitude_->setValue(drives_->item(row)->data(Qt::UserRole+1).toDouble());});
    connect(amplitude_,qOverload<double>(&QDoubleSpinBox::valueChanged),this,[this](double value){if(refreshing_||!drives_->currentItem()) return; drives_->currentItem()->setData(Qt::UserRole+1,value); updateDrives();});
    auto* logDock=new QDockWidget("Simulation log",this); auto* logPage=new QWidget; auto* ll=new QVBoxLayout(logPage); summary_=new QLabel("Place objects and ports, then generate the domain and mesh."); summary_->setWordWrap(true); ll->addWidget(summary_); log_=new QPlainTextEdit; log_->setReadOnly(true); log_->setMaximumBlockCount(2500); ll->addWidget(log_); logDock->setWidget(logPage); addDockWidget(Qt::BottomDockWidgetArea,logDock); logDock->setMaximumHeight(230);
    modelDock->setObjectName("modelDock"); propDock->setObjectName("propertiesDock"); driveDock->setObjectName("excitationDock"); logDock->setObjectName("logDock");
    coordinates_=new QLabel("Coordinates in mm"); progress_=new QProgressBar; progress_->setRange(0,1000); progress_->setMaximumWidth(250); statusBar()->addPermanentWidget(coordinates_); statusBar()->addPermanentWidget(progress_);
    connect(tabs_,&QTabWidget::currentChanged,this,[this](int index){if(index==1) field_->render(); if(index==5) monitorView_->render();});
}
QString MainWindow::sourceRoot() const { return QDir::cleanPath(QString::fromUtf8(FDTD_SOURCE_ROOT)); }
QString MainWindow::python() const {
    const auto configured=QSettings().value("python").toString(); if(!configured.isEmpty()) return configured;
    const auto env=qEnvironmentVariable("FDTD_PYTHON"); if(!env.isEmpty()) return env;
    const QStringList candidates={sourceRoot()+"/.venv/Scripts/python.exe",sourceRoot()+"/.venv/bin/python",sourceRoot()+"/venv/Scripts/python.exe",QDir::homePath()+"/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"};
    for(const auto& candidate:candidates) if(QFileInfo::exists(candidate)) return candidate;
    auto executable=QStandardPaths::findExecutable("python3"); return executable.isEmpty()?QStandardPaths::findExecutable("python"):executable;
}
void MainWindow::refresh() {
    const QSignalBlocker canvasBlocker(canvas_);
    refreshing_=true; tree_->clear(); canvas_->setProject(project_.data());
    for(const auto& category:{QString("objects"),QString("ports"),QString("sources"),QString("monitors")}) {
        auto* root=new QTreeWidgetItem(tree_,{category=="monitors"?"Field monitors":category=="objects"?"Geometry":category=="ports"?"Ports":"Plane waves"}); root->setData(0,Qt::UserRole+1,-1);
        const auto array=project_.items(category);
        for(int i=0;i<array.size();++i) { const auto object=array[i].toObject(); auto* item=new QTreeWidgetItem(root,{object["name"].toString(),category=="objects"?object["material"].toObject()["type"].toString():object["kind"].toString(),category=="sources"||category=="monitors"?QString():QString::number(object["rank"].toInt())}); item->setData(0,Qt::UserRole,category); item->setData(0,Qt::UserRole+1,i); }
        root->setExpanded(true);
    }
    const auto selected=selectedCategory_; const auto index=selectedIndex_; selectedCategory_.clear(); selectedIndex_=-1; select(selected,index);
    const int driveRow=drives_->currentRow(); drives_->clear(); const auto exc=project_.data()["excitations"].toArray();
    auto addDrive=[&](const QString& name,int mode,const QString& units){
        auto* item=new QListWidgetItem(name+":"+QString::number(mode)+"  ["+units+"]",drives_); item->setFlags(item->flags()|Qt::ItemIsUserCheckable); item->setData(Qt::UserRole,QJsonObject{{"name",name},{"mode",mode}}); item->setData(Qt::UserRole+1,1.); item->setCheckState(Qt::Unchecked);
        for(const auto& entry:exc) { const auto drive=entry.toObject(); if(drive["name"].toString()==name&&drive["mode"].toInt()==mode) { item->setCheckState(Qt::Checked); item->setData(Qt::UserRole+1,drive["amplitude"].toDouble(1)); } }
    };
    for(const auto& value:project_.items("ports")) { const auto port=value.toObject(); const bool guide=port["kind"].toString()=="waveguide"; for(int i=0;i<(guide?port["modes"].toInt(1):1);++i) addDrive(port["name"].toString(),i,guide?"sqrt(W)":"V"); }
    for(const auto& value:project_.items("sources")) addDrive(value.toObject()["name"].toString(),0,"V/m");
    if(drives_->count()) drives_->setCurrentRow(std::clamp(driveRow,0,drives_->count()-1));
    const auto s=project_.settings(); summary_->setText(QString("%1 · %2–%3 GHz · %4 ns max · %5\nDomain and PML generated from objects; NTFF always closed.").arg(s["polarization"].toString()).arg(s["f_min_ghz"].toDouble()).arg(s["f_max_ghz"].toDouble()).arg(s["max_time_ns"].toDouble()).arg(s["study"].toString()=="sparameters"?"Full S matrix":"Coherent excitations"));
    setWindowTitle(project_.data()["title"].toString("FDTD model")+"[*] — FDTD Studio"); setWindowModified(project_.dirty()); refreshing_=false;
}
void MainWindow::select(const QString& category,int index) {
    selectedCategory_=category; selectedIndex_=index; canvas_->select(category,index);
    const QSignalBlocker blocker(tree_);
    for(int r=0;r<tree_->topLevelItemCount();++r) for(int i=0;i<tree_->topLevelItem(r)->childCount();++i) { auto* item=tree_->topLevelItem(r)->child(i); if(item->data(0,Qt::UserRole).toString()==category&&item->data(0,Qt::UserRole+1).toInt()==index) { tree_->setCurrentItem(item); break; } }
    auto object=project_.item(category,index); properties_->setRowCount(object.size()); int row=0;
    for(auto i=object.begin();i!=object.end();++i) {
        QString value=i.value().isString()?i.value().toString():i.value().isDouble()?QString::number(i.value().toDouble(),'g',10):QString::fromUtf8(QJsonDocument(i.value().isObject()?QJsonDocument(i.value().toObject()):QJsonDocument(i.value().toArray())).toJson(QJsonDocument::Compact));
        properties_->setItem(row,0,new QTableWidgetItem(i.key())); properties_->setItem(row,1,new QTableWidgetItem(value)); ++row;
    }
}
void MainWindow::add(const QString& category,const QString& kind,QJsonObject initial) {
    if(running()) { statusBar()->showMessage("Stop the simulation before editing its model.",5000); return; }
    initial["kind"]=kind; initial["name"]=project_.uniqueName(kind+"_"); ObjectDialog dialog(&project_,category,initial,-1,this);
    if(dialog.exec()==QDialog::Accepted) { selectedCategory_=category; selectedIndex_=project_.items(category).size(); project_.add(category,dialog.value()); }
}
void MainWindow::editSelected() {
    if(running()||selectedIndex_<0||selectedCategory_.isEmpty()) return;
    ObjectDialog dialog(&project_,selectedCategory_,project_.item(selectedCategory_,selectedIndex_),selectedIndex_,this);
    if(dialog.exec()==QDialog::Accepted) project_.edit(selectedCategory_,selectedIndex_,dialog.value());
}
void MainWindow::deleteSelected() { if(!running()&&selectedIndex_>=0&&!selectedCategory_.isEmpty()) { const auto category=selectedCategory_; const int i=selectedIndex_; selectedIndex_=-1; project_.remove(category,i); } }
void MainWindow::settings() { if(running()) return; SettingsDialog dialog(project_.data(),this); if(dialog.exec()==QDialog::Accepted) project_.replace(dialog.value(),"Simulation settings"); }
void MainWindow::updateDrives() {
    if(refreshing_||running()) return; QJsonArray array;
    for(int i=0;i<drives_->count();++i) { auto* item=drives_->item(i); if(item->checkState()==Qt::Checked) { auto value=item->data(Qt::UserRole).toJsonObject(); value["amplitude"]=item->data(Qt::UserRole+1).toDouble(); array.append(value); } }
    auto data=project_.data(); data["excitations"]=array; project_.replace(data,"Excitation selection");
}
bool MainWindow::discardChanges() {
    if(!project_.dirty()) return true;
    const auto choice=QMessageBox::question(this,"Unsaved project","Save the current project before continuing?",QMessageBox::Save|QMessageBox::Discard|QMessageBox::Cancel);
    return choice==QMessageBox::Discard||(choice==QMessageBox::Save&&saveProject());
}
bool MainWindow::saveProject(bool as) {
    QString path=projectPath_; if(as||path.isEmpty()) path=QFileDialog::getSaveFileName(this,"Save FDTD project",path.isEmpty()?"model.fdtd.json":path,"FDTD project (*.fdtd.json)"); if(path.isEmpty()) return false;
    if(!path.endsWith(".json",Qt::CaseInsensitive)) path+=".fdtd.json";
    QString error; if(!project_.save(path,&error)) { QMessageBox::warning(this,"Save failed",error); return false; } projectPath_=path; setWindowModified(false); return true;
}
bool MainWindow::openProject(const QString& path) {
    QString error; if(!project_.load(path,&error)) { appendLog(error); return false; } projectPath_=path; canvas_->fitModel(); return true;
}
void MainWindow::chooseExample(const QString& name) { if(running()||!discardChanges()) return; if(openProject(sourceRoot()+"/gui/examples/"+name+".fdtd.json")) { projectPath_.clear(); canvas_->fitModel(); } }
void MainWindow::appendLog(const QString& text) { if(!text.trimmed().isEmpty()) log_->appendPlainText(text.trimmed()); }
void MainWindow::start(bool preview) {
    if(running()) return;
    const auto interpreter=python(); if(interpreter.isEmpty()||!QFileInfo::exists(interpreter)) { QMessageBox::warning(this,"Python unavailable","Select a Python interpreter in Simulation > Python runtime."); return; }
    runDirectory_=sourceRoot()+"/gui/runs/"+QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss")+"-"+QUuid::createUuid().toString(QUuid::Id128).left(8);
    QDir().mkpath(runDirectory_); QSaveFile file(runDirectory_+"/project.fdtd.json");
    if(!file.open(QIODevice::WriteOnly)||file.write(QJsonDocument(project_.data()).toJson())<0||!file.commit()) { appendLog("Could not write the run project: "+file.errorString()); return; }
    pending_.clear(); gotComplete_=jobCancelled_=jobFailed_=false;
    if(!preview) { results_={}; monitorView_->clear(); sPlot_->setMessage("Simulation running — S parameters will appear when the study completes."); farPlot_->setMessage("Simulation running — far fields will appear when the study completes."); }
    auto env=QProcessEnvironment::systemEnvironment(); QStringList paths{sourceRoot()}; if(QDir(sourceRoot()+"/.test-deps").exists()) paths<<sourceRoot()+"/.test-deps";
    if(!env.value("PYTHONPATH").isEmpty()) paths<<env.value("PYTHONPATH"); env.insert("PYTHONPATH",paths.join(QDir::listSeparator())); env.insert("PYTHONUNBUFFERED","1");
    worker_.setProcessEnvironment(env); worker_.setWorkingDirectory(sourceRoot());
    QStringList arguments{"-m","FDTD_2D.gui_backend","--project",runDirectory_+"/project.fdtd.json","--output",runDirectory_}; if(preview) arguments<<"--preview";
    runAction_->setEnabled(false); previewAction_->setEnabled(false); cancelAction_->setEnabled(true); progress_->setValue(0); appendLog((preview?"Generate mesh":"Run simulation")+QString(" · ")+interpreter); appendLog("Output: "+runDirectory_);
    worker_.start(interpreter,arguments); statusBar()->showMessage("Compiling geometry and ranked mesh…");
    canvas_->setEnabled(false); drives_->setEnabled(false); amplitude_->setEnabled(false);
}
void MainWindow::consume() {
    pending_+=worker_.readAllStandardOutput(); int newline;
    while((newline=pending_.indexOf('\n'))>=0) { const auto line=pending_.left(newline).trimmed(); pending_.remove(0,newline+1); if(line.isEmpty()) continue; QJsonParseError error; const auto doc=QJsonDocument::fromJson(line,&error); if(error.error==QJsonParseError::NoError&&doc.isObject()) handleEvent(doc.object()); else appendLog(QString::fromUtf8(line)); }
}
void MainWindow::loadMetadata(const QJsonObject& data) {
    metadata_=data; canvas_->setCompiled(data); modeView_->setMetadata(data);
    summary_->setText(QString("Mesh %1 × %2 · dt %3 ps · %4 scalar / %5 vector DOFs\n%6 split DOFs · %7 enlarged groups · %8 rejected anchors · %9 fallbacks").arg(data["shape"].toArray()[0].toInt()).arg(data["shape"].toArray()[1].toInt()).arg(data["dt_ps"].toDouble(),0,'g',6).arg(data["scalar_dofs"].toInt()).arg(data["vector_dofs"].toInt()).arg(data["split_dofs"].toInt()).arg(data["enlarged_groups"].toInt()).arg(data["rejected_anchors"].toArray().size()).arg(data["fallbacks"].toArray().size()));
    if(data["rejected_anchors"].toArray().size()) appendLog("Rejected anchors: "+QString::fromUtf8(QJsonDocument(data["rejected_anchors"].toArray()).toJson(QJsonDocument::Compact)));
    if(data["fallbacks"].toArray().size()) appendLog("Staircase fallback: "+QString::fromUtf8(QJsonDocument(data["fallbacks"].toArray()).toJson(QJsonDocument::Compact)));
}
void MainWindow::handleEvent(const QJsonObject& data) {
    const auto type=data["event"].toString();
    if(type=="status") { appendLog(data["message"].toString()); statusBar()->showMessage(data["message"].toString()); }
    else if(type=="compiled") {
        loadMetadata(data); fieldSnapshot_->clear(); fieldSnapshot_->addItem("Mesh · permittivity",data["field_file"].toString()); field_->load(data["field_file"].toString(),"epsilon_r",false);
        appendLog(summary_->text());
    }
    else if(type=="progress") {
        progress_->setValue(int(data["fraction"].toDouble()*1000)); statusBar()->showMessage(QString("Run %1/%2 · step %3/%4 · %5 ns").arg(data["run_index"].toInt()+1).arg(data["run_count"].toInt(1)).arg(data["step"].toInt()).arg(data["steps"].toInt()).arg(data["time"].toDouble()*1e9,0,'f',3));
        if(tabs_->currentIndex()==1) field_->load(data["field_file"].toString(),data["scalar_label"].toString());
    }
    else if(type=="complete") { gotComplete_=true; if(data["preview"].toBool()) { tabs_->setCurrentIndex(1); field_->fit(); } else { openResults(data["path"].toString()); appendLog("Results, CSV, NPZ and field snapshots saved."); } }
    else if(type=="error") { jobFailed_=true; appendLog("ERROR: "+data["message"].toString()); }
    else if(type=="cancelled") { jobCancelled_=true; appendLog(data["message"].toString()); }
}
void MainWindow::cancel() {
    if(!running()) return; jobCancelled_=true; QFile file(runDirectory_+"/cancel"); if(file.open(QIODevice::WriteOnly)) file.close(); appendLog("Stop requested.");
    const auto job=runDirectory_; QTimer::singleShot(3000,this,[this,job]{if(running()&&runDirectory_==job&&jobCancelled_) worker_.kill();});
}
bool MainWindow::openResults(const QString& path) {
    QString error; const auto data=readJson(path,&error);
    if(data["version"].toInt()!=1||!data["frequencies_ghz"].isArray()||!data["runs"].isArray()||data["runs"].toArray().isEmpty()) { appendLog("Could not open results: "+error); return false; }
    const bool same=data["project"].toObject()==project_.data(); if(!same) { project_.reset(data["project"].toObject()); projectPath_.clear(); }
    results_=data; runDirectory_=QFileInfo(path).absolutePath(); loadMetadata(data["mesh"].toObject());
    // Resolve paths relative to the selected run directory when a run was moved.
    auto resolve=[this](const QString& original){return QFileInfo::exists(original)?original:runDirectory_+"/"+QFileInfo(original).fileName();};
    { const QSignalBlocker a(incoming_),b(farRun_),c(farFrequency_),d(fieldSnapshot_);
        incoming_->clear(); for(const auto value:data["channels"].toArray()) incoming_->addItem(channel(value));
        farRun_->clear(); fieldSnapshot_->clear(); fieldSnapshot_->addItem("Mesh · permittivity",resolve(data["mesh"].toObject()["field_file"].toString()));
        const auto runs=data["runs"].toArray(); for(int i=0;i<runs.size();++i) { const auto run=runs[i].toObject(); QStringList driven; for(const auto value:run["driven_channels"].toArray()) driven<<channel(value); farRun_->addItem("Run "+QString::number(i+1)+" · "+driven.join(" + ")); fieldSnapshot_->addItem("Run "+QString::number(i+1)+" · peak energy",resolve(run["peak_field"].toString())); fieldSnapshot_->addItem("Run "+QString::number(i+1)+" · final",resolve(run["final_field"].toString())); appendLog(QString("Run %1: %2 steps, %3 ns, %4").arg(i+1).arg(run["steps"].toInt()).arg(run["time_ns"].toDouble(),0,'g',6).arg(run["stop_reason"].toString())); }
        farFrequency_->clear(); const auto frequencies=data["frequencies_ghz"].toArray(); for(const auto value:frequencies) farFrequency_->addItem(QString::number(value.toDouble(),'g',7)+" GHz"); farFrequency_->setCurrentIndex(frequencies.size()/2); if(fieldSnapshot_->count()>1) fieldSnapshot_->setCurrentIndex(1);
    }
    updateS(); updateFar(); updateField(); monitorView_->setResults(data,runDirectory_); if(!same) canvas_->fitModel(); return true;
}
void MainWindow::updateField() { if(fieldSnapshot_->currentIndex()<0) return; field_->load(fieldSnapshot_->currentData().toString(),fieldSnapshot_->currentIndex()==0?"epsilon_r":results_["scalar_label"].toString("Scalar field"),fieldSnapshot_->currentIndex()!=0); }
void MainWindow::updateS() {
    if(results_.isEmpty()||results_["s_real"].isNull()) { sPlot_->setMessage("S parameters require a full independent-port study. Coherent runs provide incoming and outgoing waves in results.json."); return; }
    const int incident=incoming_->currentIndex(); if(incident<0) return; const auto frequencies=results_["frequencies_ghz"].toArray(),real=results_["s_real"].toArray(),imag=results_["s_imag"].toArray(),labels=results_["channels"].toArray();
    QVector<Curve> curves; const int representation=sRepresentation_->currentIndex();
    for(int out=0;out<labels.size();++out) { Curve curve{channel(labels[out])+" ← "+channel(labels[incident]),{},curveColors[out%curveColors.size()]};
        for(int f=0;f<frequencies.size();++f) { const double r=numeric(real[f].toArray()[out].toArray()[incident]),i=numeric(imag[f].toArray()[out].toArray()[incident]); const std::complex<double> z(r,i); double y;
            if(!std::isfinite(r)||!std::isfinite(i)) y=std::numeric_limits<double>::quiet_NaN(); else y=representation==0?20*std::log10(std::max(std::abs(z),1e-15)):representation==1?std::arg(z)*180./3.141592653589793:representation==2?r:i;
            curve.points.append({frequencies[f].toDouble(),y}); }
        curves.append(curve);
    }
    sPlot_->setCurves(curves,"Frequency (GHz)",sRepresentation_->currentText());
}
void MainWindow::updateFar() {
    if(results_.isEmpty()||farRun_->currentIndex()<0||farFrequency_->currentIndex()<0) return;
    const auto run=results_["runs"].toArray()[farRun_->currentIndex()].toObject(); const int f=farFrequency_->currentIndex(),representation=farRepresentation_->currentIndex();
    const auto angles=results_["angles_deg"].toArray(); const auto powers=run[representation==1?"normalized_power":"power"].toArray()[f].toArray();
    const auto real=run["far_real"].toArray()[f].toArray(),imag=run["far_imag"].toArray()[f].toArray();
    double peak=0; for(const auto p:powers) if(p.isDouble()) peak=std::max(peak,p.toDouble());
    Curve curve{farRun_->currentText(),{},curveColors[0]};
    for(int a=0;a<angles.size();++a) { double y=numeric(powers[a]); if(representation==0) y=peak>0?std::max(-60.,10*std::log10(std::max(y/peak,1e-30))):std::numeric_limits<double>::quiet_NaN(); else if(representation==3) y=std::arg(std::complex<double>(numeric(real[a]),numeric(imag[a])))*180./3.141592653589793; curve.points.append({angles[a].toDouble(),y}); }
    const QString title=representation==0?"Relative power (dB)":representation==1?"W/rad per incident W":representation==2?"Raw DFT power (W s²/rad)":"Scalar phase (deg)";
    farPlot_->setCurves({curve},"Angle (deg)",title,representation!=3);
}
void MainWindow::exportPlot(Plot* plot,bool image) {
    const auto path=QFileDialog::getSaveFileName(this,image?"Export plot image":"Export plot samples",{},image?"PNG image (*.png)":"CSV data (*.csv)"); if(path.isEmpty()) return;
    if(!(image?plot->savePng(path):plot->saveCsv(path))) QMessageBox::warning(this,"Export failed","Could not write the selected file.");
}
void MainWindow::closeEvent(QCloseEvent* event) {
    if(running()) { if(QMessageBox::question(this,"Simulation running","Stop the simulation and close?",QMessageBox::Yes|QMessageBox::No)!=QMessageBox::Yes) { event->ignore(); return; } worker_.kill(); worker_.waitForFinished(3000); }
    if(!discardChanges()) { event->ignore(); return; } QSettings().setValue("geometry",saveGeometry()); QSettings().setValue("layout",saveState()); event->accept();
}
bool MainWindow::smokeImages(const QString& directory) {
    QDir().mkpath(directory); canvas_->fitModel(); tabs_->setCurrentIndex(0); QApplication::processEvents(); bool ok=grab().save(directory+"/model.png");
    tabs_->setCurrentIndex(1); field_->fit(); QApplication::processEvents(); ok=field_->savePng(directory+"/field.png")&&ok; ok=grab().save(directory+"/field-ui.png")&&ok;
    tabs_->setCurrentIndex(2); QApplication::processEvents(); ok=grab().save(directory+"/sparameters.png")&&ok;
    tabs_->setCurrentIndex(3); QApplication::processEvents(); ok=grab().save(directory+"/far-field.png")&&ok;
    tabs_->setCurrentIndex(4); QApplication::processEvents(); ok=grab().save(directory+"/port-modes.png")&&ok;
    tabs_->setCurrentIndex(5); monitorView_->fit(); QApplication::processEvents(); ok=grab().save(directory+"/frequency-fields.png")&&ok;
    if(!results_["runs"].toArray().isEmpty()&&!results_["runs"].toArray()[0].toObject()["monitors"].toArray().isEmpty()) {
        for(int phase:{0,90,180}) { ok=monitorView_->setPhase(phase)&&ok; QApplication::processEvents(); ok=monitorView_->savePng(directory+"/harmonic-"+QString::number(phase)+".png")&&ok; }
        auto* play=monitorView_->findChild<QPushButton*>("playMonitor"); auto* phase=monitorView_->findChild<QSlider*>("monitorPhase");
        const int before=phase->value(); play->setChecked(true); QEventLoop cycle;
        QTimer::singleShot(180,&cycle,&QEventLoop::quit); cycle.exec(); play->setChecked(false);
        ok=(phase->value()!=before)&&ok; ok=grab().save(directory+"/harmonic-animation.png")&&ok;
    }
    return ok;
}
