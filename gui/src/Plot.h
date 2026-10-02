#pragma once
#include <QWidget>
#include <QVector>
#include <QPointF>
#include <QColor>

struct Curve { QString name; QVector<QPointF> points; QColor color; };
class Plot : public QWidget {
    Q_OBJECT
public:
    explicit Plot(QWidget* parent=nullptr);
    void setCurves(QVector<Curve> curves,const QString& xLabel,const QString& yLabel,bool polar=false);
    void setMessage(const QString& message);
    bool saveCsv(const QString& path) const;
    bool savePng(const QString& path);
    const QVector<Curve>& curves() const { return curves_; }
    void setZeroReference(bool value) { zeroReference_=value; }
    QVector<double> yTicks() const;
protected:
    void paintEvent(QPaintEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void leaveEvent(QEvent*) override;
    void wheelEvent(QWheelEvent*) override;
    void mouseDoubleClickEvent(QMouseEvent*) override;
private:
    QVector<Curve> curves_;
    QString xLabel_,yLabel_,message_="Run a simulation or open results to inspect this plot.";
    bool polar_=false,zeroReference_=false;
    double xmin_=0,xmax_=1,ymin_=0,ymax_=1,zoom_=1.;
    QPointF cursor_{-1,-1};
    QRectF area() const;
};
