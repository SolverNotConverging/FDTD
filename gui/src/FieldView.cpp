#include "FieldView.h"
#include <QVTKOpenGLNativeWidget.h>
#include <QVBoxLayout>
#include <QFileInfo>
#include <vtkRenderer.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <vtkDataSetMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkCamera.h>
#include <vtkScalarBarActor.h>
#include <vtkLookupTable.h>
#include <vtkTextProperty.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkDoubleArray.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>
#include <vtkInteractorStyleImage.h>
#include <vtkRenderWindowInteractor.h>
#include <cmath>
#include <algorithm>

FieldView::FieldView(QWidget* parent):QWidget(parent) {
    auto* layout=new QVBoxLayout(this); layout->setContentsMargins(0,0,0,0);
    widget_=new QVTKOpenGLNativeWidget; layout->addWidget(widget_);
    window_=vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New(); renderer_=vtkSmartPointer<vtkRenderer>::New();
    window_->AddRenderer(renderer_); widget_->setRenderWindow(window_);
    auto style=vtkSmartPointer<vtkInteractorStyleImage>::New(); style->SetInteractionModeToImage2D();
    window_->GetInteractor()->SetInteractorStyle(style);
    renderer_->SetBackground(.965,.977,.988); renderer_->GetActiveCamera()->ParallelProjectionOn();
    mapper_=vtkSmartPointer<vtkDataSetMapper>::New(); mapper_->SetScalarModeToUseCellData();
    colors_=vtkSmartPointer<vtkLookupTable>::New(); colors_->SetNumberOfTableValues(256); colors_->Build(); mapper_->SetLookupTable(colors_);
    actor_=vtkSmartPointer<vtkActor>::New(); actor_->SetMapper(mapper_); actor_->GetProperty()->LightingOff();
    actor_->GetProperty()->SetEdgeColor(.24,.32,.40); actor_->GetProperty()->SetLineWidth(.7); renderer_->AddActor(actor_); actor_->SetVisibility(false);
    legend_=vtkSmartPointer<vtkScalarBarActor>::New(); legend_->SetLookupTable(colors_); legend_->SetNumberOfLabels(5); legend_->SetPosition(.84,.1); legend_->SetWidth(.12); legend_->SetHeight(.77);
    legend_->GetTitleTextProperty()->SetColor(.2,.27,.34); legend_->GetLabelTextProperty()->SetColor(.2,.27,.34); legend_->GetTitleTextProperty()->SetFontSize(13); legend_->GetLabelTextProperty()->SetFontSize(12);
    renderer_->AddViewProp(legend_); legend_->SetVisibility(false);
}
bool FieldView::load(const QString& path,const QString& title,bool signedField,const QString& array) {
    if(!QFileInfo::exists(path)) return false;
    auto reader=vtkSmartPointer<vtkXMLUnstructuredGridReader>::New(); const auto filename=path.toUtf8(); reader->SetFileName(filename.constData()); reader->Update();
    auto* data=reader->GetOutput(); if(!data||!data->GetNumberOfCells()) return false;
    if(array=="animated") {
        auto* real=data->GetCellData()->GetArray("real"); if(!real||!data->GetCellData()->GetArray("imaginary")) return false;
        auto animated=vtkSmartPointer<vtkDoubleArray>::New(); animated->DeepCopy(real); animated->SetName("animated"); data->GetCellData()->AddArray(animated);
    }
    auto* scalars=data->GetCellData()->GetArray(array.toUtf8().constData()); if(!scalars) return false;
    mapper_->SetInputData(data); mapper_->SetScalarModeToUseCellFieldData(); mapper_->SelectColorArray(array.toUtf8().constData());
    double range[2]; scalars->GetRange(range);
    if(array=="animated") { data->GetCellData()->GetArray("magnitude")->GetRange(range); range[0]=-range[1]; }
    if(signedField) { const double a=std::max({std::abs(range[0]),std::abs(range[1]),1e-30}); range[0]=-a; range[1]=a; }
    else if(range[1]<=range[0]) range[1]=range[0]+1;
    for(int i=0;i<256;++i) {
        const double t=i/255.;
        if(signedField) { const double a=t<.5?2*t:2*(1-t); colors_->SetTableValue(i,t<.5?.13+.83*a:.92,t<.5?.34+.62*a:.96-.74*(2*t-1),t<.5?.70+.26*a:.96-.73*(2*t-1),1); }
        else colors_->SetTableValue(i,.10+.75*t,.40+.23*t,.65-.42*t,1);
    }
    mapper_->SetScalarRange(range); colors_->SetRange(range); legend_->SetTitle(title.toUtf8().constData());
    actor_->SetVisibility(true); legend_->SetVisibility(true);
    if(empty_) { renderer_->ResetCamera(); renderer_->GetActiveCamera()->ParallelProjectionOn(); empty_=false; }
    render(); return true;
}
bool FieldView::setHarmonicPhase(double degrees) {
    auto* data=vtkUnstructuredGrid::SafeDownCast(mapper_->GetInput()); if(!data) return false;
    auto* real=data->GetCellData()->GetArray("real"); auto* imag=data->GetCellData()->GetArray("imaginary"); auto* animated=data->GetCellData()->GetArray("animated");
    if(!real||!imag||!animated) return false;
    const double phase=degrees*3.141592653589793/180.;
    for(vtkIdType i=0;i<real->GetNumberOfTuples();++i) animated->SetTuple1(i,real->GetTuple1(i)*std::cos(phase)-imag->GetTuple1(i)*std::sin(phase));
    animated->Modified(); data->Modified(); render(); return true;
}
void FieldView::clear() { actor_->SetVisibility(false); legend_->SetVisibility(false); empty_=true; if(isVisible()) render(); }
void FieldView::fit() { renderer_->ResetCamera(); renderer_->GetActiveCamera()->ParallelProjectionOn(); render(); }
void FieldView::setEdges(bool value) { actor_->GetProperty()->SetEdgeVisibility(value); render(); }
void FieldView::render() { if(widget_->isVisible()) window_->Render(); else widget_->update(); }
bool FieldView::savePng(const QString& path) {
    window_->Render(); auto capture=vtkSmartPointer<vtkWindowToImageFilter>::New(); capture->SetInput(window_); capture->ReadFrontBufferOff(); capture->Update();
    auto writer=vtkSmartPointer<vtkPNGWriter>::New(); writer->SetFileName(path.toUtf8().constData()); writer->SetInputConnection(capture->GetOutputPort()); writer->Write(); return QFileInfo::exists(path);
}
