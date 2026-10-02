#include "Canvas.h"
#include <QGraphicsPathItem>
#include <QGraphicsEllipseItem>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QKeyEvent>
#include <QPainter>
#include <QJsonArray>
#include <QSignalBlocker>
#include <QLineF>
#include <cmath>

namespace {
QPointF xy(const QJsonArray& p) { return {p[0].toDouble(),-p[1].toDouble()}; }
QColor color(const QJsonObject& m) {
    const auto type=m["type"].toString();
    if(type=="dielectric") return QColor("#42a5a2");
    if(type=="PMC") return QColor("#ae85d5");
    if(type=="film"||type=="thin_metal") return QColor("#df9d52");
    if(type=="SIBC") return QColor("#91a7bd");
    return QColor("#526477");
}
}
Canvas::Canvas(QWidget* parent):QGraphicsView(parent),scene_(this) {
    setScene(&scene_); setRenderHint(QPainter::Antialiasing); setDragMode(ScrollHandDrag);
    setMouseTracking(true); setTransformationAnchor(AnchorUnderMouse); setViewportUpdateMode(FullViewportUpdate);
    setBackgroundBrush(QColor("#f5f8fb")); scale(18,18);
    connect(&scene_,&QGraphicsScene::selectionChanged,this,[this]{
        const auto items=scene_.selectedItems(); if(items.isEmpty()) emit selected({},-1);
        else emit selected(items.first()->data(0).toString(),items.first()->data(1).toInt());
    });
}
void Canvas::setTool(const QString& tool) {
    tool_=tool; points_.clear(); if(sketch_) { scene_.removeItem(sketch_); delete sketch_; sketch_=nullptr; }
    setDragMode(tool=="select"?ScrollHandDrag:NoDrag); setCursor(tool=="select"?Qt::ArrowCursor:Qt::CrossCursor);
}
void Canvas::setProject(const QJsonObject& project) {
    project_=project; sketch_=nullptr; moving_=nullptr; points_.clear(); scene_.clear();
    for(const auto& category:{QString("objects"),QString("ports"),QString("monitors")}) {
        const auto array=project[category].toArray();
        for(int i=0;i<array.size();++i) {
            const auto object=array[i].toObject(); const auto kind=object["kind"].toString();
            QPainterPath path; QColor c=category=="monitors"?QColor("#ac3aae"):category=="ports"?QColor("#e14b67"):color(object["material"].toObject());
            bool area=false;
            if(kind=="rectangle"||kind=="monitor") { const double x=object["x"].toDouble(),y=object["y"].toDouble(),w=object["width"].toDouble(),h=object["height"].toDouble(); path.addRect(x,-y-h,w,h); area=kind!="monitor"; }
            else if(kind=="circle") { const double r=object["radius"].toDouble(); path.addEllipse({object["x"].toDouble(),-object["y"].toDouble()},r,r); area=true; }
            else if(kind=="polygon") { QPolygonF polygon; for(const auto p:object["vertices"].toArray()) polygon<<xy(p.toArray()); path.addPolygon(polygon); path.closeSubpath(); area=true; }
            else if(kind=="sheet"||kind=="lumped") {
                const auto start=xy(object["start"].toArray());
                if(object["end"].isArray()) { path.moveTo(start); path.lineTo(xy(object["end"].toArray())); }
                else { path.addEllipse(start,.25,.25); path.moveTo(start+QPointF(-.4,0)); path.lineTo(start+QPointF(.4,0)); path.moveTo(start+QPointF(0,-.4)); path.lineTo(start+QPointF(0,.4)); }
            }
            else if(kind=="waveguide") {
                const bool x=object["axis"].toString()=="x"; const auto span=object["span"].toArray(); const double p=object["position"].toDouble();
                const QPointF a=x?QPointF(p,-span[0].toDouble()):QPointF(span[0].toDouble(),-p);
                const QPointF b=x?QPointF(p,-span[1].toDouble()):QPointF(span[1].toDouble(),-p);
                path.moveTo(a); path.lineTo(b);
                const auto center=(a+b)/2.; const int normal=object["normal"].toInt(1);
                const QPointF direction=x?QPointF(normal*.9,0):QPointF(0,-normal*.9);
                path.moveTo(center-direction); path.lineTo(center+direction);
                const QPointF perpendicular(-direction.y(),direction.x());
                path.moveTo(center+direction); path.lineTo(center+direction*.55+perpendicular*.3);
                path.moveTo(center+direction); path.lineTo(center+direction*.55-perpendicular*.3);
            }
            auto* item=scene_.addPath(path,QPen(c,category=="ports"?2.5:1.5,Qt::SolidLine,Qt::RoundCap,Qt::RoundJoin),area?QBrush(QColor(c.red(),c.green(),c.blue(),85)):Qt::NoBrush);
            auto pen=item->pen(); pen.setCosmetic(true); item->setPen(pen);
            if(category=="monitors") { pen.setStyle(Qt::DashDotLine); item->setPen(pen); }
            item->setFlags(QGraphicsItem::ItemIsSelectable|QGraphicsItem::ItemIsMovable);
            item->setData(0,category); item->setData(1,i); item->setToolTip(object["name"].toString()+" · "+kind);
            item->setZValue(category=="monitors"?110.:category=="ports"?100.:i*.001);
        }
    }
    if(!compiled_.isEmpty()) setCompiled(compiled_);
    auto bounds=scene_.itemsBoundingRect(); if(bounds.isEmpty()) bounds=QRectF(-15,-15,30,30);
    scene_.setSceneRect(bounds.adjusted(-15,-15,15,15));
}
void Canvas::setCompiled(const QJsonObject& metadata) {
    compiled_=metadata;
    for(auto* item:scene_.items()) if(item->data(2).toBool()) { scene_.removeItem(item); delete item; }
    const auto xs=metadata["mesh_x_mm"].toArray(),ys=metadata["mesh_y_mm"].toArray();
    if(showMesh_&&xs.size()>1&&ys.size()>1) {
        QPainterPath grid;
        for(const auto x:xs) { grid.moveTo(x.toDouble(),-ys.first().toDouble()); grid.lineTo(x.toDouble(),-ys.last().toDouble()); }
        for(const auto y:ys) { grid.moveTo(xs.first().toDouble(),-y.toDouble()); grid.lineTo(xs.last().toDouble(),-y.toDouble()); }
        QPen pen(QColor(51,104,139,125),.8); pen.setCosmetic(true);
        auto* item=scene_.addPath(grid,pen); item->setData(2,true); item->setData(3,"simulationMesh");
        item->setAcceptedMouseButtons(Qt::NoButton); item->setZValue(80);
    }
    for(const auto& key:{QString("domain_mm"),QString("ntff_mm"),QString("tfsf_mm"),QString("physical_mm")}) {
        const auto b=metadata[key].toArray(); if(b.size()!=4) continue;
        QColor c=key=="domain_mm"?QColor("#7e8d9a"):key=="ntff_mm"?QColor("#167db0"):key=="tfsf_mm"?QColor("#ba8128"):QColor("#9ba5ae");
        QPen pen(c,1.3,Qt::DashLine); pen.setCosmetic(true);
        auto* item=scene_.addRect(b[0].toDouble(),-b[3].toDouble(),b[2].toDouble()-b[0].toDouble(),b[3].toDouble()-b[1].toDouble(),pen);
        item->setData(2,true); item->setZValue(-100); item->setToolTip(key);
        item->setAcceptedMouseButtons(Qt::NoButton);
    }
    scene_.setSceneRect(scene_.itemsBoundingRect().adjusted(-5,-5,5,5));
}
void Canvas::setShowMesh(bool value) { showMesh_=value; setCompiled(compiled_); }
void Canvas::select(const QString& category,int index) {
    const QSignalBlocker blocker(&scene_);
    for(auto* item:scene_.items()) if(item->flags()&QGraphicsItem::ItemIsSelectable) item->setSelected(item->data(0).toString()==category&&item->data(1).toInt()==index);
}
void Canvas::fitModel() { auto bounds=scene_.itemsBoundingRect(); if(bounds.isEmpty()) bounds=QRectF(-15,-15,30,30); fitInView(bounds.adjusted(-2,-2,2,2),Qt::KeepAspectRatio); }
QPointF Canvas::point(const QPoint& position) const {
    auto p=mapToScene(position); if(snap_>0) p={std::round(p.x()/snap_)*snap_,std::round(p.y()/snap_)*snap_}; return p;
}
void Canvas::preview(const QPointF& current) {
    if(points_.isEmpty()) return;
    if(!sketch_) { QPen pen(QColor("#168eb4"),1.8,Qt::DashLine); pen.setCosmetic(true); sketch_=scene_.addPath({},pen); sketch_->setZValue(1000); }
    QPainterPath path; const auto a=points_.first();
    if(tool_=="rectangle"||tool_=="monitor") path.addRect(QRectF(a,current).normalized());
    else if(tool_=="circle") { const double r=QLineF(a,current).length(); path.addEllipse(a,r,r); }
    else { path.moveTo(a); for(int i=1;i<points_.size();++i) path.lineTo(points_[i]); path.lineTo(current); }
    sketch_->setPath(path);
}
void Canvas::mousePressEvent(QMouseEvent* event) {
    if(tool_=="select") {
        moving_=nullptr;
        for(auto* item:items(event->position().toPoint())) if(item->flags()&QGraphicsItem::ItemIsMovable) { moving_=item; break; }
        if(moving_&&moving_->flags()&QGraphicsItem::ItemIsMovable) beforeMove_=moving_->pos(); else moving_=nullptr;
        QGraphicsView::mousePressEvent(event); return;
    }
    const auto p=point(event->position().toPoint());
    if(tool_=="polygon"&&event->button()==Qt::RightButton) {
        if(points_.size()>=3) { QVector<QPointF> result; for(const auto v:points_) result<<QPointF(v.x(),-v.y()); const auto tool=tool_; setTool("select"); emit drawn(tool,result); }
        return;
    }
    if(event->button()!=Qt::LeftButton) return;
    if(tool_=="lumped"&&project_["settings"].toObject()["polarization"].toString()=="TM") {
        setTool("select"); emit drawn("lumped",{{p.x(),-p.y()}}); return;
    }
    if(tool_=="polygon") points_<<p; else points_={p}; preview(p);
}
void Canvas::mouseMoveEvent(QMouseEvent* event) {
    const auto p=point(event->position().toPoint()); emit coordinate(p.x(),-p.y());
    if(tool_=="select") QGraphicsView::mouseMoveEvent(event); else preview(p);
}
void Canvas::mouseReleaseEvent(QMouseEvent* event) {
    if(tool_=="select") {
        QGraphicsView::mouseReleaseEvent(event);
        if(moving_) { const auto delta=moving_->pos()-beforeMove_; const auto category=moving_->data(0).toString(); const int index=moving_->data(1).toInt(); moving_=nullptr; if(QLineF({},delta).length()>1e-8) emit moved(category,index,delta.x(),-delta.y()); }
        return;
    }
    if(tool_=="polygon"||points_.isEmpty()||event->button()!=Qt::LeftButton) return;
    const auto p=point(event->position().toPoint()); const auto a=points_.first();
    if(QLineF(a,p).length()<1e-6) return;
    const auto kind=tool_; setTool("select"); emit drawn(kind,{{a.x(),-a.y()},{p.x(),-p.y()}});
}
void Canvas::wheelEvent(QWheelEvent* event) { const double factor=std::pow(1.0015,event->angleDelta().y()); scale(factor,factor); }
void Canvas::keyPressEvent(QKeyEvent* event) { if(event->key()==Qt::Key_Escape) setTool("select"); else QGraphicsView::keyPressEvent(event); }
void Canvas::drawBackground(QPainter* painter,const QRectF& rect) {
    QGraphicsView::drawBackground(painter,rect);
    const double pixels=std::abs(transform().m11()); double spacing=1.;
    while(spacing*pixels<35) spacing*=2.; while(spacing*pixels>110) spacing/=2.;
    QPen pen(QColor("#e1e8ef"),0); painter->setPen(pen);
    for(double x=std::floor(rect.left()/spacing)*spacing;x<rect.right();x+=spacing) painter->drawLine(QPointF(x,rect.top()),QPointF(x,rect.bottom()));
    for(double y=std::floor(rect.top()/spacing)*spacing;y<rect.bottom();y+=spacing) painter->drawLine(QPointF(rect.left(),y),QPointF(rect.right(),y));
    painter->setPen(QPen(QColor("#b9c8d5"),0)); painter->drawLine(QPointF(0,rect.top()),QPointF(0,rect.bottom())); painter->drawLine(QPointF(rect.left(),0),QPointF(rect.right(),0));
}
