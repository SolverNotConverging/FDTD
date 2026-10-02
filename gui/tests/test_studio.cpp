#include <QtTest>
#include <QTemporaryDir>
#include <QFile>
#include "Project.h"
#include "Canvas.h"
#include "Plot.h"
#include "Inspection.h"
#include "Ribbon.h"
#include "Radiation.h"
#include "SnapshotPlayer.h"
#include "Dialogs.h"
#include <QDialogButtonBox>
#include <QPushButton>
#include <QToolButton>
#include <QActionGroup>
#include <QGraphicsPathItem>
#include <QComboBox>
#include <limits>
#include <cmath>

namespace {
QJsonArray oneRow(const QJsonArray& values) {QJsonArray rows; rows.append(values); return rows;}
QJsonObject radiationFixture() {
    constexpr double pi=3.14159265358979323846;
    QJsonArray powers; for(int i=0;i<5;++i) powers.append(1/(2*pi));
    return {{"angles_deg",QJsonArray{0,45,120,250,360}},{"frequencies_ghz",QJsonArray{30}},
        {"channels",QJsonArray{QJsonArray{"feed",0},QJsonArray{"feed",1},QJsonArray{"load",0}}},
        {"runs",QJsonArray{QJsonObject{{"driven_channels",oneRow({"feed",0})},{"power",oneRow(powers)},
            {"incoming_real",oneRow({2,0,0})},{"incoming_imag",oneRow({0,0,0})},
            {"outgoing_real",oneRow({1,.5,.9})},{"outgoing_imag",oneRow({0,0,0})}}}}};
}
void replaceRadiationRun(QJsonObject& result,const QJsonObject& run) {result["runs"]=QJsonArray{run};}
}

class StudioTests : public QObject {
    Q_OBJECT
private slots:
    void timePlaybackIsChronologicalAndStaysWithinOneRun() {
        SnapshotPlayer player; player.setSnapshots({{"Late","late","Hz",0,30,.3},{"Early","early","Hz",0,10,.1},{"Other","other","Hz",1,20,.2}});
        auto* selector=player.selector(); auto* play=player.findChild<QPushButton*>("playTimeFields"); QVERIFY(play->isEnabled());
        QCOMPARE(selector->currentData().toString(),QString("late")); player.advanceFrame(); QCOMPARE(selector->currentData().toString(),QString("early")); player.advanceFrame(); QCOMPARE(selector->currentData().toString(),QString("late"));
        play->setChecked(true); QVERIFY(play->isChecked()); selector->setCurrentIndex(2); QVERIFY(!play->isEnabled()); QVERIFY(!play->isChecked());
        selector->setCurrentIndex(0); play->setChecked(true); player.clear(); QVERIFY(!play->isChecked()); QCOMPARE(selector->count(),0);
    }
    void sParameterMagnitudeAlwaysHasAZeroTick() {
        Plot plot; plot.setZeroReference(true);
        for(const auto& values:{QVector<QPointF>{{1,-40},{2,-3}},QVector<QPointF>{{1,-3},{2,1.2}},QVector<QPointF>{{1,2},{2,3}}}) {
            plot.setCurves({Curve{"S",values,QColor("blue")}},"GHz","Magnitude (dB)"); QVERIFY(plot.yTicks().contains(0.));
        }
    }
    void uniformModalFieldDoesNotMagnifyRoundoff() {
        Plot plot; plot.setCurves({Curve{"TEM",{{0,21.34},{1,21.34+1e-12}},QColor("blue")}},"mm","Hz");
        const auto ticks=plot.yTicks(); QVERIFY(ticks.last()-ticks.first()>1.);
    }
    void inwardNormalAndPolarizationPreserveLegacyFields() {
        Project model; auto guide=QJsonObject{{"name","guide"},{"kind","waveguide"},{"normal",-1},{"span",QJsonArray{0,12}}}; model.add("ports",guide);
        ObjectDialog dialog(&model,"ports",guide,0); QCOMPARE(dialog.findChild<QComboBox*>("inwardNormal")->currentText(),QString("+1"));
        dialog.findChild<QDialogButtonBox*>()->button(QDialogButtonBox::Ok)->click(); QCOMPARE(dialog.value()["inward_normal"].toInt(),1); QVERIFY(!dialog.value().contains("normal"));
        auto project=model.data(); auto settings=project["settings"].toObject(); settings["polarization"]="TM"; project["settings"]=settings;
        SettingsDialog physics(project); auto* polarization=physics.findChild<QComboBox*>("polarization"); QCOMPARE(polarization->currentText(),QString("Ez"));
        physics.findChild<QDialogButtonBox*>()->button(QDialogButtonBox::Ok)->click(); QCOMPARE(physics.value()["settings"].toObject()["polarization"].toString(),QString("TM"));
    }
    void radiationUsesClosedCircleAndCorrectFeedPower() {
        const auto result=radiationFixture(); const auto metrics=radiationMetrics(result,0,0);
        QVERIFY(std::abs(metrics.radiatedPower-1)<1e-12); QCOMPARE(metrics.incidentPower,4.); QCOMPARE(metrics.acceptedPower,2.75);
        // Isotropic radiation is 0 dB in 2D, even with nonuniform angular samples.
        for(int i=0;i<5;++i) {
            QVERIFY(std::abs(metrics.directivity[i]-1)<1e-12);
            QVERIFY(std::abs(metrics.gain[i]-1/2.75)<1e-12); QVERIFY(std::abs(metrics.realizedGain[i]-.25)<1e-12);
        }
        // Receiving-port power is a load; mode conversion at a feed is reflection.
        auto coherent=result; auto run=coherent["runs"].toArray()[0].toObject();
        run["driven_channels"]=QJsonArray{QJsonArray{"feed",0},QJsonArray{"feed",1},QJsonArray{"load",0}};
        run["incoming_real"]=oneRow({2,0,3}); replaceRadiationRun(coherent,run);
        const auto combined=radiationMetrics(coherent,0,0);
        QCOMPARE(combined.incidentPower,13.); QVERIFY(std::abs(combined.acceptedPower-10.94)<1e-12);
        QVERIFY(std::abs(combined.gain[0]-1/10.94)<1e-12); QVERIFY(std::abs(combined.realizedGain[0]-1/13.)<1e-12);
    }
    void radiationRejectsUndefinedAndWeakNormalization() {
        auto result=radiationFixture(); auto run=result["runs"].toArray()[0].toObject();
        run["driven_channels"]=QJsonArray{QJsonArray{"plane",0},QJsonArray{"feed",0}}; replaceRadiationRun(result,run);
        auto metrics=radiationMetrics(result,0,0); QVERIFY(std::isfinite(metrics.directivity[0])); QVERIFY(std::isnan(metrics.gain[0])); QVERIFY(!metrics.realizedGainUnavailable.isEmpty());
        result=radiationFixture(); run=result["runs"].toArray()[0].toObject();
        run["outgoing_real"]=oneRow({2,0,0}); replaceRadiationRun(result,run);
        metrics=radiationMetrics(result,0,0); QCOMPARE(metrics.acceptedPower,0.); QVERIFY(std::isnan(metrics.gain[0])); QVERIFY(std::isfinite(metrics.realizedGain[0]));
        result=radiationFixture(); result["frequencies_ghz"]=QJsonArray{30,31}; run=result["runs"].toArray()[0].toObject();
        run["power"]=QJsonArray{run["power"].toArray()[0],run["power"].toArray()[0]};
        run["incoming_real"]=QJsonArray{QJsonArray{2,0,0},QJsonArray{1e-8,0,0}}; run["incoming_imag"]=QJsonArray{QJsonArray{0,0,0},QJsonArray{0,0,0}};
        run["outgoing_real"]=QJsonArray{QJsonArray{0,0,0},QJsonArray{0,0,0}}; run["outgoing_imag"]=run["incoming_imag"]; replaceRadiationRun(result,run);
        metrics=radiationMetrics(result,0,1); QVERIFY(std::isfinite(metrics.directivity[0])); QVERIFY(std::isnan(metrics.gain[0])); QVERIFY(std::isnan(metrics.realizedGain[0]));
        result=radiationFixture(); result["angles_deg"]=QJsonArray{0,45,120,250,350}; metrics=radiationMetrics(result,0,0);
        QVERIFY(std::isnan(metrics.directivity[0])); QVERIFY(!metrics.directivityUnavailable.isEmpty());
        result=radiationFixture(); run=result["runs"].toArray()[0].toObject(); run["power"]=oneRow({1,1,QJsonValue(),1,1}); replaceRadiationRun(result,run);
        metrics=radiationMetrics(result,0,0); QVERIFY(std::isnan(metrics.directivity[0])); QVERIFY(std::isnan(metrics.realizedGain[0]));
    }
    void radiationExcludesNonpropagatingFeedModes() {
        auto result=radiationFixture(); auto run=result["runs"].toArray()[0].toObject();
        result["mesh"]=QJsonObject{{"port_modes",QJsonArray{QJsonObject{{"name","feed"},{"modes",QJsonArray{
            QJsonObject{{"index",0},{"valid",QJsonArray{true}}},QJsonObject{{"index",1},{"valid",QJsonArray{false}}}}}}}}};
        run["incoming_real"]=oneRow({2,QJsonValue(),0}); run["incoming_imag"]=oneRow({0,QJsonValue(),0});
        run["outgoing_real"]=oneRow({1,QJsonValue(),.9}); run["outgoing_imag"]=oneRow({0,QJsonValue(),0}); replaceRadiationRun(result,run);
        auto metrics=radiationMetrics(result,0,0); QCOMPARE(metrics.acceptedPower,3.); QVERIFY(std::abs(metrics.gain[0]-1/3.)<1e-12);
        run["driven_channels"]=oneRow({"feed",1}); replaceRadiationRun(result,run); metrics=radiationMetrics(result,0,0);
        QVERIFY(std::isnan(metrics.realizedGain[0])); QVERIFY(!metrics.realizedGainUnavailable.isEmpty());
    }
    void ribbonSharesActionStateAndKeepsPagesIndependent() {
        Ribbon ribbon; const int model=ribbon.addPage("Model"),simulation=ribbon.addPage("Simulation");
        auto* shapes=ribbon.addGroup(model,"Shapes"); auto* solver=ribbon.addGroup(simulation,"Solver");
        QAction select("Select"),rectangle("Rectangle"),run("Run"),stop("Stop");
        QActionGroup drawing(&ribbon); drawing.setExclusive(true);
        select.setCheckable(true); rectangle.setCheckable(true); drawing.addAction(&select); drawing.addAction(&rectangle); select.setChecked(true);
        auto* selectButton=shapes->addCommand(&select,"Select","select"); auto* rectangleButton=shapes->addCommand(&rectangle,"Rectangle","rectangle");
        auto* runButton=solver->addCommand(&run,"Run\nF5","run"); auto* stopButton=solver->addCommand(&stop,"Stop","stop"); stop.setEnabled(false);
        ribbon.resize(640,150); ribbon.show(); QTest::qWait(10);
        QTest::mouseClick(rectangleButton,Qt::LeftButton); QVERIFY(rectangle.isChecked()); QVERIFY(rectangleButton->isChecked()); QVERIFY(!selectButton->isChecked());
        QSignalSpy triggered(&run,&QAction::triggered); ribbon.setCurrentPage(simulation); QCOMPARE(triggered.count(),0); QVERIFY(!stopButton->isEnabled());
        QTest::mouseClick(runButton,Qt::LeftButton); QCOMPARE(triggered.count(),1);
        run.setEnabled(false); stop.setEnabled(true); QVERIFY(!runButton->isEnabled()); QVERIFY(stopButton->isEnabled()); QCOMPARE(runButton->text(),QString("Run\nF5"));
        ribbon.setCurrentPage(model); QVERIFY(rectangle.isChecked()); QCOMPARE(triggered.count(),1);
    }
    void projectUndoRoundTrip() {
        Project model; QTemporaryDir folder;
        QJsonObject circle{{"name","circle"},{"kind","circle"},{"x",2.},{"y",3.},{"radius",1.},{"rank",40},{"material",QJsonObject{{"type","PEC"}}}};
        model.add("objects",circle); QVERIFY(model.dirty()); QCOMPARE(model.items("objects").size(),1);
        model.move("objects",0,4.,-2.); QCOMPARE(model.item("objects",0)["x"].toDouble(),6.);
        model.undoStack()->undo(); QCOMPARE(model.item("objects",0)["x"].toDouble(),2.);
        model.undoStack()->redo(); QString error; QVERIFY(model.save(folder.path()+"/model.fdtd.json",&error)); QVERIFY(!model.dirty());
        Project restored; QVERIFY(restored.load(folder.path()+"/model.fdtd.json",&error)); QCOMPARE(restored.data(),model.data());
        model.undoStack()->undo(); QVERIFY(model.dirty());
    }
    void deletionRemovesDriveAndUndoRestoresIt() {
        Project model; auto data=model.data(); data["ports"]=QJsonArray{QJsonObject{{"name","feed"},{"kind","lumped"},{"start",QJsonArray{0.,0.}}}};
        data["excitations"]=QJsonArray{QJsonObject{{"name","feed"},{"mode",0},{"amplitude",1.}}}; model.reset(data);
        model.remove("ports",0); QCOMPARE(model.items("excitations").size(),0); model.undoStack()->undo(); QCOMPARE(model.items("excitations").size(),1);
    }
    void polygonAndGuideMoveInPhysicalCoordinates() {
        Project model; model.add("objects",QJsonObject{{"name","triangle"},{"kind","polygon"},{"vertices",QJsonArray{QJsonArray{0,0},QJsonArray{1,0},QJsonArray{0,1}}}});
        model.move("objects",0,3,5); QCOMPARE(model.item("objects",0)["vertices"].toArray()[2].toArray(),(QJsonArray{3,6}));
        model.add("ports",QJsonObject{{"name","guide"},{"kind","waveguide"},{"axis","y"},{"position",2},{"span",QJsonArray{0,12}}});
        model.move("ports",0,4,7); QCOMPARE(model.item("ports",0)["position"].toDouble(),9.); QCOMPARE(model.item("ports",0)["span"].toArray(),(QJsonArray{4,16}));
    }
    void corruptProjectDoesNotReplaceCurrentModel() {
        Project model; const auto before=model.data(); QTemporaryDir folder; QFile file(folder.path()+"/bad.json"); QVERIFY(file.open(QIODevice::WriteOnly)); file.write("{broken"); file.close(); QString error;
        QVERIFY(!model.load(file.fileName(),&error)); QCOMPARE(model.data(),before); QVERIFY(!error.isEmpty());
    }
    void plotExportRetainsInvalidGaps() {
        Plot plot; QTemporaryDir folder; plot.resize(640,400);
        const double nan=std::numeric_limits<double>::quiet_NaN(); plot.setCurves({Curve{"S11",{{1,-2},{2,nan},{3,-5}},QColor("blue")}},"GHz","dB");
        QVERIFY(plot.saveCsv(folder.path()+"/plot.csv")); QFile file(folder.path()+"/plot.csv"); QVERIFY(file.open(QIODevice::ReadOnly)); const auto contents=file.readAll().replace("\r\n","\n"); QVERIFY(contents.contains("\"S11\",2,\n")); QVERIFY(!contents.contains("nan")); QVERIFY(plot.savePng(folder.path()+"/plot.png"));
    }
    void namesAreGlobalAndDeterministic() {
        Project model; model.add("objects",QJsonObject{{"name","item1"}}); model.add("ports",QJsonObject{{"name","item2"}});
        QCOMPARE(model.uniqueName("item"),QString("item3")); QVERIFY(model.nameExists("item1")); QVERIFY(!model.nameExists("item1","objects",0));
    }
    void renamingAndReducingModesKeepsDriveReferencesValid() {
        Project model; auto snapshot=model.data();
        snapshot["ports"]=QJsonArray{QJsonObject{{"name","guide"},{"kind","waveguide"},{"modes",2}}};
        snapshot["excitations"]=QJsonArray{QJsonObject{{"name","guide"},{"mode",0},{"amplitude",2}},QJsonObject{{"name","guide"},{"mode",1},{"amplitude",1}}};
        model.reset(snapshot); auto port=model.item("ports",0); port["name"]="renamed"; port["modes"]=1; model.edit("ports",0,port);
        QCOMPARE(model.items("excitations").size(),1); QCOMPARE(model.items("excitations")[0].toObject()["name"].toString(),QString("renamed"));
        model.undoStack()->undo(); QCOMPARE(model.data(),snapshot);
    }
    void canvasDrawingEmitsMillimetreCoordinates() {
        Canvas canvas; canvas.resize(700,500); canvas.setProject(Project::empty()); canvas.setSnap(0); canvas.show();
        QTest::qWait(10); QString kind; QVector<QPointF> points;
        connect(&canvas,&Canvas::drawn,this,[&](const QString& value,const QVector<QPointF>& xy){kind=value;points=xy;});
        canvas.setTool("rectangle"); const auto a=canvas.mapFromScene(QPointF(1,-2)),b=canvas.mapFromScene(QPointF(4,-6));
        QTest::mousePress(canvas.viewport(),Qt::LeftButton,Qt::NoModifier,a);
        QTest::mouseMove(canvas.viewport(),b); QTest::mouseRelease(canvas.viewport(),Qt::LeftButton,Qt::NoModifier,b);
        QCOMPARE(kind,QString("rectangle")); QCOMPARE(points.size(),2);
        QVERIFY(std::abs(points[0].x()-1)<.1); QVERIFY(std::abs(points[0].y()-2)<.1);
        QVERIFY(std::abs(points[1].x()-4)<.1); QVERIFY(std::abs(points[1].y()-6)<.1);
    }
    void simulationMeshUsesNonuniformNodesAndCanBeHidden() {
        Canvas canvas; canvas.setProject(Project::empty());
        canvas.setCompiled(QJsonObject{{"mesh_x_mm",QJsonArray{0.,.25,2.}},{"mesh_y_mm",QJsonArray{-1.,.5,4.}}});
        auto mesh=[&]()->QGraphicsPathItem* {for(auto* item:canvas.scene()->items()) if(item->data(3).toString()=="simulationMesh") return qgraphicsitem_cast<QGraphicsPathItem*>(item); return nullptr;};
        QVERIFY(mesh()); QCOMPARE(mesh()->path().elementCount(),12); QCOMPARE(mesh()->path().elementAt(2).x,.25);
        QCOMPARE(mesh()->path().elementAt(8).y,-.5); QVERIFY(!(mesh()->flags()&QGraphicsItem::ItemIsMovable));
        canvas.setShowMesh(false); QVERIFY(!mesh()); canvas.setShowMesh(true); QVERIFY(mesh());
    }
    void monitorsMovePersistAndUndoWithoutBecomingMaterials() {
        Project model; QJsonObject monitor{{"kind","monitor"},{"name","region"},{"x",1},{"y",2},{"width",5},{"height",6},{"frequencies_ghz",QJsonArray{26,30,34}}};
        model.add("monitors",monitor); QVERIFY(model.nameExists("region")); model.move("monitors",0,3,-2);
        QCOMPARE(model.item("monitors",0)["x"].toDouble(),4.); model.undoStack()->undo(); QCOMPARE(model.item("monitors",0),monitor);
        QTemporaryDir folder; QString error; QVERIFY(model.save(folder.path()+"/monitor.json",&error)); Project restored; QVERIFY(restored.load(folder.path()+"/monitor.json",&error)); QCOMPARE(restored.items("monitors"),model.items("monitors"));
    }
    void trackedModeSelectorsUseEachFrequencyAndComponent() {
        PortModeView view;
        QJsonObject mode{{"index",0},{"valid",QJsonArray{true,true}},{"overlap",QJsonArray{1,.98}},
            {"beta_real_rad_m",QJsonArray{100,200}},{"beta_imag_rad_m",QJsonArray{0,-2}},{"attenuation_np_m",QJsonArray{0,2}},{"effective_index",QJsonArray{.2,.3}},
            {"q_real",QJsonArray{QJsonArray{1,2},QJsonArray{3,4}}},{"q_imag",QJsonArray{QJsonArray{0,0},QJsonArray{0,0}}},
            {"p_real",QJsonArray{QJsonArray{.1,.2},QJsonArray{.3,.4}}},{"p_imag",QJsonArray{QJsonArray{0,0},QJsonArray{0,0}}}};
        QJsonObject port{{"name","feed"},{"axis","x"},{"scalar_label","Ez"},{"scalar_unit","V/m"},{"tangent_label","Ht"},{"tangent_unit","A/m"},
            {"frequencies_ghz",QJsonArray{26,34}},{"transverse_mm",QJsonArray{.5,1.5}},{"modes",QJsonArray{mode}}};
        view.setMetadata(QJsonObject{{"port_modes",QJsonArray{port}}});
        auto* anchors=view.findChild<QComboBox*>("modeFrequency"); QCOMPARE(anchors->count(),2);
        auto* field=view.findChild<Plot*>("modeFieldPlot"); QCOMPARE(field->curves()[0].points[1].y(),4.);
        anchors->setCurrentIndex(0); QCOMPARE(field->curves()[0].points[1].y(),2.);
        auto* dispersion=view.findChild<Plot*>("modeDispersionPlot"); QCOMPARE(dispersion->curves()[0].points[1].y(),200.);
        view.setMetadata({}); QCOMPARE(anchors->count(),0); QVERIFY(field->curves().isEmpty());
    }
};
QTEST_MAIN(StudioTests)
#include "test_studio.moc"
