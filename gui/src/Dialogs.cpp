#include "Dialogs.h"
#include "Project.h"
#include <QFormLayout>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDialogButtonBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QScrollArea>
#include <QTabWidget>
#include <QMessageBox>
#include <QLabel>
#include <QRegularExpression>
#include <cmath>

ObjectDialog::ObjectDialog(Project* project,const QString& category,QJsonObject initial,int index,QWidget* parent)
    :QDialog(parent),project_(project),category_(category),initial_(std::move(initial)),index_(index) {
    setWindowTitle(index<0?"Add model item":"Edit model item"); resize(490,720);
    auto* layout=new QVBoxLayout(this); auto* identity=new QFormLayout;
    name_=new QLineEdit(initial_["name"].toString(project->uniqueName(category=="objects"?"object":"port"))); identity->addRow("Name",name_);
    kind_=new QComboBox;
    kind_->addItems(category=="objects"?QStringList{"rectangle","circle","polygon","sheet"}:category=="ports"?QStringList{"lumped","waveguide"}:category=="monitors"?QStringList{"monitor"}:QStringList{"plane"});
    kind_->setCurrentText(initial_["kind"].toString()); identity->addRow("Type",kind_); layout->addLayout(identity);
    auto* scroll=new QScrollArea; scroll->setWidgetResizable(true); fields_=new QWidget; form_=new QFormLayout(fields_); scroll->setWidget(fields_); layout->addWidget(scroll);
    auto* buttons=new QDialogButtonBox(QDialogButtonBox::Ok|QDialogButtonBox::Cancel); layout->addWidget(buttons);
    connect(buttons,&QDialogButtonBox::accepted,this,&ObjectDialog::finish); connect(buttons,&QDialogButtonBox::rejected,this,&QDialog::reject);
    connect(kind_,&QComboBox::currentTextChanged,this,&ObjectDialog::rebuild); rebuild();
}
void ObjectDialog::number(const QString& key,const QString& title,double fallback,double minimum,double maximum,int decimals) {
    auto* box=new QDoubleSpinBox; box->setDecimals(decimals); box->setRange(minimum,maximum); box->setValue(initial_[key].toDouble(fallback)); box->setKeyboardTracking(false); box->setSingleStep(decimals==0?1.:.1); form_->addRow(title,box); numbers_[key]=box;
}
void ObjectDialog::choice(const QString& key,const QString& title,const QStringList& values,const QString& fallback) {
    auto* combo=new QComboBox; combo->addItems(values); combo->setCurrentText(initial_[key].toString(fallback)); form_->addRow(title,combo); choices_[key]=combo;
}
double ObjectDialog::n(const QString& key) const { return numbers_.value(key)->value(); }
QString ObjectDialog::c(const QString& key) const { return choices_.value(key)->currentText(); }
void ObjectDialog::rebuild() {
    while(form_->rowCount()) form_->removeRow(0); numbers_.clear(); choices_.clear(); vertices_=nullptr; material_=nullptr; frequencies_=nullptr;
    const auto kind=kind_->currentText();
    auto pointNumbers=[&](const QString& prefix,const QJsonArray& point,double x,double y) {
        if(point.size()==2) { initial_[prefix+"x"]=point[0]; initial_[prefix+"y"]=point[1]; }
        number(prefix+"x",prefix+"X (mm)",x); number(prefix+"y",prefix+"Y (mm)",y);
    };
    if(kind=="rectangle"||kind=="circle"||kind=="monitor") { number("x",kind=="circle"?"Center X (mm)":"Minimum X (mm)",0); number("y",kind=="circle"?"Center Y (mm)":"Minimum Y (mm)",0); }
    if(kind=="rectangle"||kind=="monitor") { number("width","Width (mm)",5,.000001); number("height","Height (mm)",5,.000001); }
    if(kind=="monitor") {
        QStringList values; for(const auto f:initial_["frequencies_ghz"].toArray()) values<<QString::number(f.toDouble(),'g',10);
        if(values.isEmpty()) values<<QString::number((project_->settings()["f_min_ghz"].toDouble()+project_->settings()["f_max_ghz"].toDouble())/2.);
        frequencies_=new QLineEdit(values.join(", ")); form_->addRow("Monitor frequencies (GHz)",frequencies_);
        form_->addRow(new QLabel("Rectangular regional Ez (TM) or Hz (TE) DFT.\nComma-separated increasing frequencies; up to 201.\nThe region must fit inside the generated domain."));
    }
    if(kind=="circle") number("radius","Radius (mm)",3,.000001);
    if(kind=="polygon") {
        vertices_=new QPlainTextEdit; QString text; for(const auto p:initial_["vertices"].toArray()) { const auto xy=p.toArray(); text+=QString::number(xy[0].toDouble())+", "+QString::number(xy[1].toDouble())+"\n"; }
        if(text.isEmpty()) text="0, 0\n5, 0\n3, 4"; vertices_->setPlainText(text); vertices_->setMinimumHeight(150); form_->addRow("Vertices: x, y in mm",vertices_);
    }
    if(kind=="sheet"||kind=="lumped") {
        pointNumbers("start_",initial_["start"].toArray(),0,0);
        if(kind=="sheet"||project_->settings()["polarization"].toString()=="TE") pointNumbers("end_",initial_["end"].toArray(),0,5);
        if(kind=="lumped") number("resistance","Load resistance (ohm)",50,.000001);
    }
    if(kind=="waveguide") {
        choice("axis","Normal axis",{"x","y"},"x"); number("position","Plane position (mm)",0);
        auto span=initial_["span"].toArray(); if(span.size()==2) { initial_["span_min"]=span[0]; initial_["span_max"]=span[1]; }
        number("span_min","Aperture start (mm)",0); number("span_max","Aperture end (mm)",12);
        initial_["normal_sign"]=initial_["normal"].toInt(1)==1?"+1":"-1"; choice("normal_sign","Outward normal",{"+1","-1"},"+1");
        number("modes","Tracked modes",1,1,16,0); number("mesh_step_mm","Port mesh step (mm; 0 = auto)",0,0,1e6);
        number("length_cells","Virtual guide length (cells)",32,5,1000,0); number("pml_cells","Virtual guide PML (cells)",12,2,500,0); number("clearance_cells","Source clearance (cells)",6,1,500,0);
    }
    if(category_=="ports") number("depth_mm","Invariant extrusion depth (mm)",1,.000001);
    if(category_=="sources") { choice("axis","Propagation axis",{"x","y"},"x"); initial_["direction_sign"]=initial_["direction"].toInt(1)==1?"+1":"-1"; choice("direction_sign","Direction",{"+1","-1"},"+1"); }
    if(category_!="sources"&&category_!="monitors") number("rank","Mesh anchor rank",category_=="ports"?100:40,-1000000,1000000,0);
    if(category_=="objects") {
        const auto m=initial_["material"].toObject(); material_=new QComboBox;
        material_->addItems({"PEC","PMC","dielectric","SIBC","film","thin_metal"}); material_->setCurrentText(m["type"].toString("PEC")); form_->addRow("Material",material_);
        for(const auto& key:{QString("epsilon_r"),QString("mu_r"),QString("sigma_e"),QString("sigma_m"),QString("resistance"),QString("conductivity"),QString("thickness_um")}) if(m.contains(key)) initial_[key]=m[key];
        number("epsilon_r","Relative permittivity",1,1); number("mu_r","Relative permeability",1,1);
        number("sigma_e","Electric conductivity (S/m)",0,0,1e12); number("sigma_m","Magnetic loss",0,0,1e12);
        number("resistance","Surface / sheet resistance (ohm)",50,0,1e12);
        initial_["surface_model"]=m["model"].toString("resistance"); choice("surface_model","SIBC model",{"resistance","conductor"},"resistance");
        number("conductivity","Metal conductivity (S/m)",5.8e7,.000001,1e12); number("thickness_um","Thin metal thickness (um)",1,.000001);
        form_->addRow(new QLabel("Films require a zero-thickness sheet. Coordinates are always millimetres."));
        auto visibility=[this]{
            const auto type=material_->currentText(); const bool dielectric=type=="dielectric",surface=type=="SIBC",film=type=="film",thin=type=="thin_metal";
            for(const auto& key:{QString("epsilon_r"),QString("mu_r"),QString("sigma_e"),QString("sigma_m")}) form_->setRowVisible(numbers_[key],dielectric);
            form_->setRowVisible(choices_["surface_model"],surface); form_->setRowVisible(numbers_["resistance"],film||(surface&&c("surface_model")=="resistance"));
            form_->setRowVisible(numbers_["conductivity"],thin||(surface&&c("surface_model")=="conductor")); form_->setRowVisible(numbers_["thickness_um"],thin);
        };
        connect(material_,&QComboBox::currentTextChanged,this,visibility); connect(choices_["surface_model"],&QComboBox::currentTextChanged,this,visibility); visibility();
    }
}
void ObjectDialog::finish() {
    const auto name=name_->text().trimmed(),kind=kind_->currentText();
    auto error=[this](const QString& text){QMessageBox::warning(this,"Invalid model item",text);};
    if(name.isEmpty()||project_->nameExists(name,category_,index_)) { error("Use a unique, nonempty name."); return; }
    QJsonObject value{{"name",name},{"kind",kind}};
    if(category_!="sources"&&category_!="monitors") value["rank"]=int(n("rank"));
    if(kind=="rectangle"||kind=="circle"||kind=="monitor") { value["x"]=n("x"); value["y"]=n("y"); }
    if(kind=="rectangle"||kind=="monitor") { value["width"]=n("width"); value["height"]=n("height"); }
    if(kind=="monitor") {
        QJsonArray samples; double previous=0;
        for(const auto& text:frequencies_->text().split(QRegularExpression("[,;\\s]+"),Qt::SkipEmptyParts)) {
            bool ok=false; const double f=text.toDouble(&ok);
            if(!ok||!std::isfinite(f)||f<=previous) { error("Monitor frequencies must be positive and increasing."); return; }
            samples.append(f); previous=f;
        }
        if(samples.isEmpty()||samples.size()>201) { error("Supply 1–201 monitor frequencies."); return; } value["frequencies_ghz"]=samples;
    }
    if(kind=="circle") value["radius"]=n("radius");
    if(kind=="polygon") {
        QJsonArray vertices; const auto lines=vertices_->toPlainText().split('\n',Qt::SkipEmptyParts);
        for(const auto& line:lines) { const auto words=line.trimmed().split(QRegularExpression("[,;\\s]+"),Qt::SkipEmptyParts); bool okx=false,oky=false;
            if(words.size()!=2) { error("Each vertex needs two numbers: x, y."); return; }
            const double x=words[0].toDouble(&okx),y=words[1].toDouble(&oky); if(!okx||!oky||!std::isfinite(x)||!std::isfinite(y)) { error("Invalid polygon coordinate."); return; } vertices.append(QJsonArray{x,y});
        }
        if(vertices.size()<3) { error("A polygon needs at least three vertices."); return; } value["vertices"]=vertices;
    }
    if(kind=="sheet"||kind=="lumped") {
        value["start"]=QJsonArray{n("start_x"),n("start_y")};
        if(numbers_.contains("end_x")) { value["end"]=QJsonArray{n("end_x"),n("end_y")};
            if(value["start"]==value["end"]) { error("The line must have nonzero length."); return; }
            if(kind=="lumped"&&n("start_x")!=n("end_x")&&n("start_y")!=n("end_y")) { error("A TE terminal must be axis-aligned."); return; }
        }
        if(kind=="lumped") value["resistance"]=n("resistance");
    }
    if(kind=="waveguide") {
        if(n("span_max")<=n("span_min")||n("length_cells")<n("pml_cells")+n("clearance_cells")+3) { error("Check aperture ordering and guide length (PML + clearance + 3 minimum)."); return; }
        value["axis"]=c("axis"); value["position"]=n("position"); value["span"]=QJsonArray{n("span_min"),n("span_max")}; value["normal"]=c("normal_sign")=="+1"?1:-1;
        for(const auto& key:{QString("modes"),QString("length_cells"),QString("pml_cells"),QString("clearance_cells")}) value[key]=int(n(key));
        if(n("mesh_step_mm")>0) value["mesh_step_mm"]=n("mesh_step_mm");
    }
    if(category_=="ports") value["depth_mm"]=n("depth_mm");
    if(category_=="sources") { value["axis"]=c("axis"); value["direction"]=c("direction_sign")=="+1"?1:-1; }
    if(material_) {
        const auto type=material_->currentText();
        if((type=="film"||type=="thin_metal")&&kind!="sheet") { error("Transmissive films need sheet geometry."); return; }
        if(kind=="sheet"&&type=="dielectric") { error("Sheets need PEC, PMC, SIBC or a thin-film model."); return; }
        QJsonObject m{{"type",type}};
        if(type=="dielectric") for(const auto& key:{QString("epsilon_r"),QString("mu_r"),QString("sigma_e"),QString("sigma_m")}) m[key]=n(key);
        if(type=="film"||type=="SIBC") m["resistance"]=n("resistance");
        if(type=="SIBC") m["model"]=c("surface_model");
        if(type=="thin_metal"||type=="SIBC") m["conductivity"]=n("conductivity");
        if(type=="thin_metal") m["thickness_um"]=n("thickness_um"); value["material"]=m;
    }
    result_=value; accept();
}

SettingsDialog::SettingsDialog(const QJsonObject& project,QWidget* parent):QDialog(parent),result_(project) {
    setWindowTitle("Simulation and meshing settings"); resize(570,650); auto* layout=new QVBoxLayout(this); auto* tabs=new QTabWidget; layout->addWidget(tabs);
    auto settings=project["settings"].toObject(); auto* physics=new QWidget; auto* mesh=new QWidget; auto* run=new QWidget;
    auto* pf=new QFormLayout(physics); auto* mf=new QFormLayout(mesh); auto* rf=new QFormLayout(run);
    tabs->addTab(physics,"Physics"); tabs->addTab(mesh,"Mesh & domain"); tabs->addTab(run,"Run & stopping");
    polarization_=new QComboBox; polarization_->addItems({"TM","TE"}); polarization_->setCurrentText(settings["polarization"].toString("TM")); pf->addRow("Polarization",polarization_);
    study_=new QComboBox; study_->addItem("Full S matrix (independent port drives)","sparameters"); study_->addItem("Selected coherent excitations","excitation"); study_->setCurrentIndex(settings["study"].toString()=="excitation"?1:0); pf->addRow("Study",study_);
    auto add=[&](QFormLayout* f,const QString& key,const QString& title,double low,double high,int decimals=6) { auto* box=new QDoubleSpinBox; box->setRange(low,high); box->setDecimals(decimals); box->setValue(settings[key].toDouble()); box->setKeyboardTracking(false); f->addRow(title,box); numbers_[key]=box; };
    add(pf,"f_min_ghz","Start frequency (GHz)",.000001,1e9); add(pf,"f_max_ghz","Stop frequency (GHz)",.000001,1e9); add(pf,"frequency_count","DFT samples",1,2001,0);
    add(pf,"pulse_ghz","Pulse carrier (GHz)",0,1e9); add(pf,"pulse_width_ps","Gaussian width (ps)",.000001,1e9); add(pf,"pulse_delay_ps","Pulse delay (ps; 0 = auto)",0,1e9);
    const auto bg=project["background"].toObject(); settings["background_epsilon"]=bg["epsilon_r"].toDouble(1); settings["background_mu"]=bg["mu_r"].toDouble(1);
    add(pf,"background_epsilon","Background permittivity",1,1e6); add(pf,"background_mu","Background permeability",1,1e6);
    add(mf,"max_step_mm","Maximum cell size (mm)",.000001,1e6); add(mf,"min_step_mm","Minimum cell size (mm)",.000001,1e6);
    add(mf,"min_dt_ps","Minimum CFL dt (ps; 0 = none)",0,1e6); add(mf,"cells_per_wavelength","Cells per wavelength",4,200,0); add(mf,"growth","Adjacent growth limit",1.01,10);
    add(mf,"enlargement","Minimum retained area fraction",0,1); add(mf,"pml_cells","Domain PML cells",2,100,0); add(mf,"clearance_wavelengths","Background clearance (wavelengths)",0,10);
    mf->addRow(new QLabel("The domain, closed NTFF and optional TF/SF boxes follow the objects."));
    add(rf,"max_time_ns","Maximum duration (ns)",.000001,1e9); add(rf,"min_time_ns","Minimum duration (ns)",0,1e9);
    add(rf,"field_tolerance","Field energy tolerance (0 = off)",0,.999999,9); add(rf,"dft_tolerance","DFT tolerance (0 = off)",0,.999999,9);
    rf->addRow(new QLabel("All ports receive; checked excitations transmit in a coherent run.\nS-matrix studies drive each channel independently.\nBoth stopping tests must pass when enabled."));
    auto* buttons=new QDialogButtonBox(QDialogButtonBox::Ok|QDialogButtonBox::Cancel); layout->addWidget(buttons); connect(buttons,&QDialogButtonBox::accepted,this,&SettingsDialog::finish); connect(buttons,&QDialogButtonBox::rejected,this,&QDialog::reject);
}
void SettingsDialog::finish() {
    const auto n=[this](const QString& key){return numbers_[key]->value();};
    if(n("f_max_ghz")<n("f_min_ghz")||n("max_step_mm")<n("min_step_mm")||n("min_time_ns")>n("max_time_ns")||(n("f_max_ghz")>n("f_min_ghz")&&n("frequency_count")<2)) {
        QMessageBox::warning(this,"Invalid settings","Check frequency ordering, cell sizes, sample count and run durations."); return;
    }
    auto settings=result_["settings"].toObject();
    for(auto i=numbers_.cbegin();i!=numbers_.cend();++i) if(!i.key().startsWith("background_")) settings[i.key()]=i.value()->value();
    settings["polarization"]=polarization_->currentText(); settings["study"]=study_->currentData().toString(); result_["settings"]=settings;
    auto bg=result_["background"].toObject(); bg["epsilon_r"]=n("background_epsilon"); bg["mu_r"]=n("background_mu"); result_["background"]=bg;
    accept();
}
