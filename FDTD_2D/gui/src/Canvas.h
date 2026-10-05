#pragma once
#include <QGraphicsView>
#include <QJsonObject>
#include <QVector>

class Canvas : public QGraphicsView {
    Q_OBJECT
public:
    explicit Canvas(QWidget* parent=nullptr);
    void setProject(const QJsonObject& project);
    void setTool(const QString& tool);
    void setCompiled(const QJsonObject& metadata);
    void setShowMesh(bool value);
    void select(const QString& category,int index);
    void fitModel();
    void setSnap(double value) { snap_=value; }
signals:
    void selected(const QString& category,int index);
    void drawn(const QString& kind,const QVector<QPointF>& points);
    void moved(const QString& category,int index,double dx,double dy);
    void coordinate(double x,double y);
protected:
    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;
    void wheelEvent(QWheelEvent*) override;
    void keyPressEvent(QKeyEvent*) override;
    void drawBackground(QPainter*,const QRectF&) override;
private:
    QGraphicsScene scene_;
    QString tool_="select";
    QJsonObject project_,compiled_;
    QVector<QPointF> points_;
    QGraphicsPathItem* sketch_=nullptr;
    QGraphicsItem* moving_=nullptr;
    QPointF beforeMove_;
    double snap_=.1;
    bool showMesh_=true;
    QPointF point(const QPoint& position) const;
    void preview(const QPointF& current);
};
