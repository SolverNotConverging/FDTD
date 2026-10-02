#pragma once
#include <QWidget>
#include <vtkSmartPointer.h>
class vtkRenderer;
class vtkGenericOpenGLRenderWindow;
class vtkActor;
class vtkDataSetMapper;
class vtkScalarBarActor;
class vtkLookupTable;
class QVTKOpenGLNativeWidget;

class FieldView : public QWidget {
    Q_OBJECT
public:
    explicit FieldView(QWidget* parent=nullptr);
    bool load(const QString& path,const QString& title,bool signedField=true,const QString& array="field");
    bool setHarmonicPhase(double degrees);
    void clear(); void fit(); void setEdges(bool value);
    bool savePng(const QString& path);
    void render();
private:
    QVTKOpenGLNativeWidget* widget_;
    vtkSmartPointer<vtkRenderer> renderer_;
    vtkSmartPointer<vtkGenericOpenGLRenderWindow> window_;
    vtkSmartPointer<vtkActor> actor_;
    vtkSmartPointer<vtkDataSetMapper> mapper_;
    vtkSmartPointer<vtkScalarBarActor> legend_;
    vtkSmartPointer<vtkLookupTable> colors_;
    bool empty_=true;
};
