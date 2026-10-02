#include "Project.h"
#include <QFile>
#include <QSaveFile>
#include <QJsonDocument>
#include <QUndoCommand>

class ProjectEdit : public QUndoCommand {
public:
    ProjectEdit(Project* project,QJsonObject before,QJsonObject after,const QString& title)
        :QUndoCommand(title),p(project),oldValue(std::move(before)),newValue(std::move(after)){}
    void undo() override { p->apply(oldValue); }
    void redo() override { p->apply(newValue); }
private:
    Project* p; QJsonObject oldValue,newValue;
};

Project::Project(QObject* parent):QObject(parent),data_(empty()),undo_(this){}
QJsonObject Project::empty() {
    return QJsonObject{{"version",1},{"units","mm"},{"title","Untitled model"},
        {"objects",QJsonArray{}},{"ports",QJsonArray{}},{"sources",QJsonArray{}},{"excitations",QJsonArray{}},{"monitors",QJsonArray{}},
        {"background",QJsonObject{{"epsilon_r",1.},{"mu_r",1.}}},
        {"settings",QJsonObject{{"polarization","TE"},{"study","sparameters"},{"pulse_mode","auto"},{"time_snapshot_interval_ns",0.},{"f_min_ghz",10.},
            {"f_max_ghz",20.},{"frequency_count",41},{"max_step_mm",1.},{"min_step_mm",.2},
            {"min_dt_ps",0.},{"cells_per_wavelength",20},{"growth",2.},{"pml_cells",8},
            {"clearance_wavelengths",.25},{"max_time_ns",5.},{"min_time_ns",0.},
            {"pulse_ghz",15.},{"pulse_width_ps",60.},{"pulse_delay_ps",0.},
            {"field_tolerance",1e-5},{"dft_tolerance",1e-4},{"enlargement",.3},{"fallback",true}}}};
}
void Project::apply(const QJsonObject& value) { data_=value; emit changed(); }
void Project::reset(const QJsonObject& value) { undo_.clear(); data_=value; undo_.setClean(); emit changed(); }
void Project::replace(const QJsonObject& value,const QString& title) {
    if(value!=data_) undo_.push(new ProjectEdit(this,data_,value,title));
}
QJsonObject Project::item(const QString& category,int index) const {
    const auto array=items(category); return index>=0&&index<array.size()?array[index].toObject():QJsonObject{};
}
void Project::add(const QString& category,const QJsonObject& value) {
    auto copy=data_; auto array=items(category); array.append(value); copy[category]=array;
    replace(copy,"Add "+value["name"].toString());
}
void Project::edit(const QString& category,int index,const QJsonObject& value) {
    auto array=items(category); if(index<0||index>=array.size()) return;
    const auto oldName=array[index].toObject()["name"].toString();
    auto copy=data_; array[index]=value; copy[category]=array;
    if(category=="ports"||category=="sources") {
        auto drives=copy["excitations"].toArray();
        for(int i=drives.size()-1;i>=0;--i) { auto drive=drives[i].toObject();
            if(drive["name"].toString()!=oldName) continue;
            const int modes=category=="ports"&&value["kind"].toString()=="waveguide"?value["modes"].toInt(1):1;
            if(drive["mode"].toInt()>=modes) { drives.removeAt(i); continue; }
            drive["name"]=value["name"]; drives[i]=drive;
        }
        copy["excitations"]=drives;
    }
    replace(copy,"Edit "+value["name"].toString());
}
void Project::remove(const QString& category,int index) {
    auto array=items(category); if(index<0||index>=array.size()) return;
    const auto name=array[index].toObject()["name"].toString(); auto copy=data_;
    array.removeAt(index); copy[category]=array;
    auto drives=copy["excitations"].toArray();
    for(int i=drives.size()-1;i>=0;--i) if(drives[i].toObject()["name"].toString()==name) drives.removeAt(i);
    copy["excitations"]=drives; replace(copy,"Delete "+name);
}
void Project::move(const QString& category,int index,double dx,double dy) {
    auto value=item(category,index); if(value.isEmpty()||(dx==0&&dy==0)) return;
    auto translate=[&](const QJsonArray& point){return QJsonArray{point[0].toDouble()+dx,point[1].toDouble()+dy};};
    const auto kind=value["kind"].toString();
    if(kind=="rectangle"||kind=="circle"||kind=="monitor") { value["x"]=value["x"].toDouble()+dx; value["y"]=value["y"].toDouble()+dy; }
    else if(kind=="polygon") { QJsonArray vertices; for(const auto p:value["vertices"].toArray()) vertices.append(translate(p.toArray())); value["vertices"]=vertices; }
    else if(kind=="sheet"||kind=="lumped") { value["start"]=translate(value["start"].toArray()); if(value["end"].isArray()) value["end"]=translate(value["end"].toArray()); }
    else if(kind=="waveguide") {
        const bool x=value["axis"].toString()=="x";
        value["position"]=value["position"].toDouble()+(x?dx:dy);
        const auto span=value["span"].toArray(); value["span"]=QJsonArray{span[0].toDouble()+(x?dy:dx),span[1].toDouble()+(x?dy:dx)};
    }
    edit(category,index,value);
}
bool Project::nameExists(const QString& name,const QString& category,int except) const {
    for(const auto& group:{QString("objects"),QString("ports"),QString("sources"),QString("monitors")}) {
        const auto array=items(group);
        for(int i=0;i<array.size();++i) if(!(group==category&&i==except)&&array[i].toObject()["name"].toString()==name) return true;
    }
    return false;
}
QString Project::uniqueName(const QString& prefix) const {
    int i=1; while(nameExists(prefix+QString::number(i))) ++i; return prefix+QString::number(i);
}
bool Project::load(const QString& path,QString* error) {
    QFile file(path); if(!file.open(QIODevice::ReadOnly)) { *error=file.errorString(); return false; }
    QJsonParseError parse; const auto doc=QJsonDocument::fromJson(file.readAll(),&parse);
    const auto root=doc.object();
    if(parse.error!=QJsonParseError::NoError||!doc.isObject()||root["version"].toInt()!=1||root["units"].toString()!="mm"
       ||!root["objects"].isArray()||!root["ports"].isArray()||!root["settings"].isObject()) {
        *error="Expected a version 1 FDTD project with millimetre coordinates."; return false;
    }
    reset(root); return true;
}
bool Project::save(const QString& path,QString* error) {
    QSaveFile file(path);
    if(!file.open(QIODevice::WriteOnly)||file.write(QJsonDocument(data_).toJson(QJsonDocument::Indented))<0||!file.commit()) {
        *error=file.errorString(); return false;
    }
    undo_.setClean(); return true;
}
