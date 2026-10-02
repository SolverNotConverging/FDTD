#include "Plot.h"
#include <QPainter>
#include <QPainterPath>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QToolTip>
#include <QSaveFile>
#include <QTextStream>
#include <limits>
#include <cmath>
#include <algorithm>

Plot::Plot(QWidget* parent):QWidget(parent) { setMinimumSize(350,250); setMouseTracking(true); setAutoFillBackground(true); }
QRectF Plot::area() const { return QRectF(82,38,width()-116,height()-102); }
void Plot::setMessage(const QString& value) { curves_.clear(); message_=value; update(); }
void Plot::setCurves(QVector<Curve> curves,const QString& x,const QString& y,bool polar) {
    curves_=std::move(curves); xLabel_=x; yLabel_=y; polar_=polar; zoom_=1;
    xmin_=ymin_=std::numeric_limits<double>::infinity(); xmax_=ymax_=-xmin_;
    for(const auto& curve:curves_) for(const auto& p:curve.points) if(std::isfinite(p.x())&&std::isfinite(p.y())) {
        xmin_=std::min(xmin_,p.x()); xmax_=std::max(xmax_,p.x()); ymin_=std::min(ymin_,p.y()); ymax_=std::max(ymax_,p.y());
    }
    if(!std::isfinite(xmin_)) { curves_.clear(); message_="No valid samples for this selection."; }
    else {
        if(xmax_<=xmin_) { xmin_-=.5; xmax_+=.5; }
        if(ymax_<=ymin_) { ymin_-=1.; ymax_+=1.; }
        const double pad=(ymax_-ymin_)*.08;
        if(polar_) { if(yLabel_.contains("dB")) { ymin_=-60.; ymax_=0.; } else { ymin_=0.; ymax_=std::max(ymax_,1e-300); } }
        else { ymin_-=pad; ymax_+=pad; }
    }
    update();
}
void Plot::paintEvent(QPaintEvent*) {
    QPainter painter(this); painter.setRenderHint(QPainter::Antialiasing); painter.fillRect(rect(),QColor("#ffffff"));
    if(curves_.isEmpty()) { painter.setPen(QColor("#657487")); painter.drawText(rect().adjusted(40,40,-40,-40),Qt::AlignCenter|Qt::TextWordWrap,message_); return; }
    const auto plot=area(); painter.setFont(QFont("Segoe UI",9));
    if(polar_) {
        const auto center=plot.center(); const double radius=std::min(plot.width(),plot.height())/2.-12;
        for(int i=1;i<=4;++i) { painter.setPen(QColor("#dce5ed")); painter.drawEllipse(center,radius*i/4.,radius*i/4.); painter.setPen(QColor("#718096")); painter.drawText(center+QPointF(4,-radius*i/4.),QString::number(ymin_+(ymax_-ymin_)*i/4.,'g',3)); }
        constexpr double pi=3.14159265358979323846;
        for(int degrees=0;degrees<360;degrees+=30) { const double angle=degrees*pi/180.; const QPointF unit(std::cos(angle),-std::sin(angle)); painter.setPen(QColor("#e2e8f0")); painter.drawLine(center,center+unit*radius); painter.setPen(QColor("#536276")); const auto label=center+unit*(radius+18); painter.drawText(QRectF(label-QPointF(22,10),QSizeF(44,20)),Qt::AlignCenter,QString::number(degrees)+QChar(0x00b0)); }
        for(const auto& curve:curves_) { QPainterPath path; bool begun=false; for(const auto& point:curve.points) { if(!std::isfinite(point.y())) { begun=false; continue; } const double r=std::clamp((point.y()-ymin_)/(ymax_-ymin_),0.,1.)*radius; const double a=point.x()*pi/180.; const auto p=center+QPointF(std::cos(a),-std::sin(a))*r; if(begun) path.lineTo(p); else path.moveTo(p); begun=true; } painter.setPen(QPen(curve.color,2)); painter.drawPath(path); }
        painter.setPen(QColor("#526477")); painter.drawText(QRectF(0,height()-35,width(),25),Qt::AlignCenter,yLabel_);
    } else {
        const double center=(xmin_+xmax_)/2.,half=(xmax_-xmin_)/2./zoom_; const double lo=center-half,hi=center+half;
        for(int i=0;i<=5;++i) { const double x=plot.left()+plot.width()*i/5.,y=plot.bottom()-plot.height()*i/5.; painter.setPen(QColor("#e5ebf1")); painter.drawLine(QPointF(x,plot.top()),QPointF(x,plot.bottom())); painter.drawLine(QPointF(plot.left(),y),QPointF(plot.right(),y)); painter.setPen(QColor("#5c6d80")); painter.drawText(QRectF(x-40,plot.bottom()+7,80,22),Qt::AlignCenter,QString::number(lo+(hi-lo)*i/5.,'g',5)); painter.drawText(QRectF(2,y-11,70,22),Qt::AlignRight|Qt::AlignVCenter,QString::number(ymin_+(ymax_-ymin_)*i/5.,'g',5)); }
        painter.setPen(QColor("#a8b6c5")); painter.drawRect(plot);
        painter.save(); painter.setClipRect(plot.adjusted(-1,-1,1,1));
        for(const auto& curve:curves_) { QPainterPath path; bool begun=false; for(const auto& point:curve.points) { if(!std::isfinite(point.y())) { begun=false; continue; } const QPointF p(plot.left()+(point.x()-lo)/(hi-lo)*plot.width(),plot.bottom()-(point.y()-ymin_)/(ymax_-ymin_)*plot.height()); if(begun) path.lineTo(p); else path.moveTo(p); begun=true; } painter.setPen(QPen(curve.color,2)); painter.drawPath(path); }
        if(plot.contains(cursor_)) { painter.setPen(QPen(QColor("#98a9b9"),1,Qt::DashLine)); painter.drawLine(QPointF(cursor_.x(),plot.top()),QPointF(cursor_.x(),plot.bottom())); }
        painter.restore(); painter.setPen(QColor("#526477")); painter.drawText(QRectF(plot.left(),height()-35,plot.width(),25),Qt::AlignCenter,xLabel_); painter.save(); painter.translate(19,plot.center().y()); painter.rotate(-90); painter.drawText(QRectF(-plot.height()/2.,-12,plot.height(),25),Qt::AlignCenter,yLabel_); painter.restore();
    }
    int x=86; const int y=18;
    for(const auto& curve:curves_) { painter.setPen(QPen(curve.color,2)); painter.drawLine(x,y,x+16,y); painter.setPen(QColor("#34465a")); painter.drawText(x+22,y+4,curve.name); x+=painter.fontMetrics().horizontalAdvance(curve.name)+52; }
}
void Plot::mouseMoveEvent(QMouseEvent* event) {
    cursor_=event->position();
    if(!curves_.isEmpty()&&area().contains(cursor_)) {
        const auto plot=area(); const double center=(xmin_+xmax_)/2.,half=(xmax_-xmin_)/2./zoom_;
        double target=center-half+(cursor_.x()-plot.left())/plot.width()*2*half;
        if(polar_) { const auto delta=cursor_-plot.center(); target=std::atan2(-delta.y(),delta.x())*180./3.14159265358979323846; if(target<0) target+=360.; }
        QString tip=xLabel_+": "+QString::number(target,'g',6);
        for(const auto& curve:curves_) {
            const QPointF* nearest=nullptr; double best=std::numeric_limits<double>::infinity();
            for(const auto& point:curve.points) { double distance=std::abs(point.x()-target); if(polar_) distance=std::min(distance,360.-distance); if(std::isfinite(point.y())&&distance<best) { nearest=&point; best=distance; } }
            if(nearest) tip+="\n"+curve.name+": "+QString::number(nearest->y(),'g',6);
        }
        QToolTip::showText(event->globalPosition().toPoint(),tip,this);
    }
    update();
}
void Plot::leaveEvent(QEvent*) { cursor_={-1,-1}; update(); }
void Plot::wheelEvent(QWheelEvent* event) { if(!polar_) { zoom_=std::clamp(zoom_*std::pow(1.001,event->angleDelta().y()),1.,50.); update(); } }
void Plot::mouseDoubleClickEvent(QMouseEvent*) { zoom_=1.; update(); }
bool Plot::savePng(const QString& path) { return grab().save(path,"PNG"); }
bool Plot::saveCsv(const QString& path) const {
    QSaveFile file(path); if(!file.open(QIODevice::WriteOnly|QIODevice::Text)) return false;
    QTextStream stream(&file); stream<<"curve,"<<xLabel_<<","<<yLabel_<<"\n";
    for(const auto& curve:curves_) for(const auto& p:curve.points) stream<<'"'<<QString(curve.name).replace('"',"\"\"")<<"\","<<QString::number(p.x(),'g',16)<<","<<(std::isfinite(p.y())?QString::number(p.y(),'g',16):QString())<<"\n";
    stream.flush(); return file.commit();
}
