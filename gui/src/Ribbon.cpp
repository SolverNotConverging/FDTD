#include "Ribbon.h"
#include <QAction>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QToolButton>
#include <QTabBar>
#include <QStackedWidget>
#include <QScrollArea>
#include <QPainter>
#include <QPainterPath>

RibbonGroup::RibbonGroup(const QString& title, QWidget* parent):QWidget(parent) {
    setObjectName("ribbonGroup");
    auto* layout=new QVBoxLayout(this); layout->setContentsMargins(8,4,8,2); layout->setSpacing(2);
    commands_=new QHBoxLayout; commands_->setSpacing(2); layout->addLayout(commands_,1);
    auto* caption=new QLabel(title); caption->setObjectName("ribbonCaption"); caption->setAlignment(Qt::AlignCenter); layout->addWidget(caption);
    setSizePolicy(QSizePolicy::Fixed,QSizePolicy::Expanding);
}
QToolButton* RibbonGroup::addCommand(QAction* action,const QString& label,const QString& glyph) {
    if(action->icon().isNull()) action->setIcon(Ribbon::icon(glyph));
    auto* button=new QToolButton(this); button->setDefaultAction(action); button->setText(label);
    button->setToolButtonStyle(Qt::ToolButtonTextUnderIcon); button->setIconSize(QSize(32,32));
    button->setFixedSize(84,78); button->setAutoRaise(true);
    // QAction changes update enabled/checked/icon state; retain the compact ribbon label.
    auto refresh=[button,action,label]{
        button->setText(label);
        button->setToolTip(action->toolTip()+(action->shortcut().isEmpty()?QString():" ("+action->shortcut().toString()+")"));
    };
    connect(action,&QAction::changed,button,refresh); refresh();
    commands_->addWidget(button); return button;
}
void RibbonGroup::addControl(QWidget* control) { commands_->addWidget(control,0,Qt::AlignVCenter); }

Ribbon::Ribbon(QWidget* parent):QWidget(parent) {
    setObjectName("commandRibbon"); setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Fixed);
    auto* layout=new QVBoxLayout(this); layout->setContentsMargins(0,0,0,0); layout->setSpacing(0);
    auto* strip=new QWidget; strip->setObjectName("ribbonStrip"); auto* row=new QHBoxLayout(strip); row->setContentsMargins(12,0,12,0); row->setSpacing(0);
    tabs_=new QTabBar; tabs_->setObjectName("ribbonTabs"); tabs_->setExpanding(false); tabs_->setDrawBase(false); row->addWidget(tabs_);
    row->addStretch(); auto* name=new QLabel("FDTD STUDIO"); name->setObjectName("ribbonBrand"); row->addWidget(name);
    layout->addWidget(strip); pages_=new QStackedWidget; pages_->setObjectName("ribbonPages"); pages_->setFixedHeight(112); layout->addWidget(pages_);
    connect(tabs_,&QTabBar::currentChanged,pages_,&QStackedWidget::setCurrentIndex);
    setStyleSheet(
        "QWidget#commandRibbon{background:#f5f6f8;border-bottom:1px solid #b8c2ce;}"
        "QWidget#ribbonStrip{background:#17619b;}"
        "QLabel#ribbonBrand{color:#d8e9f6;font-size:10px;font-weight:600;letter-spacing:2px;}"
        "QTabBar#ribbonTabs::tab{background:transparent;color:white;padding:7px 22px;border:none;min-width:64px;}"
        "QTabBar#ribbonTabs::tab:selected{background:#f5f6f8;color:#204968;}"
        "QTabBar#ribbonTabs::tab:hover:!selected{background:#2b76ad;}"
        "QWidget#ribbonGroup{border-right:1px solid #ced5de;background:transparent;}"
        "QLabel#ribbonCaption{color:#5e6976;font-size:11px;}"
        "QToolButton{color:#25384a;border:1px solid transparent;border-radius:3px;padding:3px;background:transparent;}"
        "QToolButton:hover{background:#e0edf9;border-color:#8ab7de;}"
        "QToolButton:checked{background:#d4e7f8;border-color:#75a7d0;}"
        "QToolButton:pressed{background:#bdd9f0;} QToolButton:disabled{color:#96a0aa;}"
        "QScrollArea{border:none;background:#f5f6f8;} QScrollBar:horizontal{height:6px;}"
    );
}
int Ribbon::addPage(const QString& title) {
    auto* scroll=new QScrollArea; scroll->setWidgetResizable(true); scroll->setFrameShape(QFrame::NoFrame);
    scroll->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff); scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    auto* content=new QWidget; auto* row=new QHBoxLayout(content); row->setContentsMargins(4,0,4,0); row->setSpacing(0); row->addStretch();
    scroll->setWidget(content); const int page=pages_->addWidget(scroll); tabs_->addTab(title); return page;
}
RibbonGroup* Ribbon::addGroup(int page,const QString& title) {
    auto* scroll=qobject_cast<QScrollArea*>(pages_->widget(page)); auto* row=qobject_cast<QHBoxLayout*>(scroll->widget()->layout());
    auto* group=new RibbonGroup(title,scroll->widget()); row->insertWidget(row->count()-1,group); return group;
}
void Ribbon::setCurrentPage(int page) { tabs_->setCurrentIndex(page); }
int Ribbon::currentPage() const { return tabs_->currentIndex(); }

QIcon Ribbon::icon(const QString& name) {
    // Original vector-style glyphs painted at high DPI; no proprietary assets.
    QPixmap pixels(64,64); pixels.fill(Qt::transparent); QPainter p(&pixels); p.setRenderHint(QPainter::Antialiasing); p.scale(2,2);
    const QColor blue("#2876b4"),teal("#258b8c"),gold("#dfa33b"),red("#c75850");
    p.setPen(QPen(blue,1.7,Qt::SolidLine,Qt::RoundCap,Qt::RoundJoin)); p.setBrush(QColor("#d9e9f6"));
    if(name=="rectangle") p.drawRoundedRect(QRectF(4,7,24,18),1,1);
    else if(name=="circle") p.drawEllipse(QRectF(5,5,22,22));
    else if(name=="polygon") p.drawPolygon(QPolygonF{{4,24},{8,6},{21,3},{28,16},{21,28}});
    else if(name=="sheet") {p.setPen(QPen(gold,3)); p.drawLine(QPointF(5,26),QPointF(27,6)); p.setPen(QPen(blue,1)); p.drawLine(QPointF(7,27),QPointF(29,7));}
    else if(name=="select") {p.setBrush(blue);p.drawPolygon(QPolygonF{{7,3},{7,25},{13,19},{18,29},{22,27},{17,17},{26,16}});}
    else if(name=="fit") {p.drawRect(QRectF(10,10,12,12)); for(int i=0;i<4;++i){p.save();p.translate(16,16);p.rotate(90*i);p.drawLine(QPointF(7,7),QPointF(12,12));p.drawLine(QPointF(7,12),QPointF(12,12));p.drawLine(QPointF(12,7),QPointF(12,12));p.restore();}}
    else if(name=="mesh") {p.drawRect(QRectF(3,3,26,26));p.setPen(QPen(teal,1));for(int x:{8,13,16,18,23})p.drawLine(x,3,x,29);for(int y:{8,12,17,24})p.drawLine(3,y,29,y);}
    else if(name=="run") {p.setBrush(teal);p.setPen(Qt::NoPen);p.drawPolygon(QPolygonF{{8,4},{28,16},{8,28}});}
    else if(name=="stop") {p.setBrush(red);p.setPen(Qt::NoPen);p.drawRoundedRect(QRectF(6,6,21,21),2,2);}
    else if(name=="waveguide"||name=="lumped") {p.drawRect(QRectF(5,4,22,24));p.setPen(QPen(gold,3));p.drawLine(7,6,7,26);p.setPen(QPen(teal,1.8));if(name=="waveguide"){QPainterPath path;path.moveTo(9,24);path.cubicTo(18,24,16,8,25,8);p.drawPath(path);}else{p.drawLine(17,6,17,13);p.drawLine(13,13,21,13);p.drawLine(13,17,21,17);p.drawLine(17,17,17,26);}}
    else if(name=="plane"||name=="modes") {p.setBrush(Qt::NoBrush);for(int y:{7,15,23}){QPainterPath path;path.moveTo(2,y);path.cubicTo(8,y-7,9,y+7,16,y);path.cubicTo(23,y-7,24,y+7,30,y);p.drawPath(path);}}
    else if(name=="monitor"||name=="fields") {p.drawRect(QRectF(3,5,26,22));p.setPen(Qt::NoPen);for(int x=0;x<5;++x)for(int y=0;y<4;++y){p.setBrush(QColor::fromHsv(210-(x+y)*25,160,220));p.drawRect(QRectF(5+x*4.5,7+y*4.5,4.5,4.5));}}
    else if(name=="plot"||name=="far") {p.setBrush(Qt::NoBrush);p.setPen(QPen(blue,1.5));p.drawLine(4,3,4,28);p.drawLine(4,28,30,28);p.setPen(QPen(teal,2));QPainterPath path;if(name=="plot"){path.moveTo(6,10);path.cubicTo(13,10,12,25,17,25);path.cubicTo(23,25,22,7,29,7);}else{path.addEllipse(QRectF(12,4,14,21));p.drawLine(6,26,20,6);}p.drawPath(path);}
    else if(name=="edit") {p.drawRect(QRectF(4,5,19,23));p.setPen(QPen(gold,4));p.drawLine(12,23,28,7);}
    else if(name=="delete") {p.setPen(QPen(red,2));p.drawLine(6,7,26,7);p.drawRect(QRectF(9,9,14,19));p.drawLine(12,3,20,3);for(int x:{13,19})p.drawLine(x,13,x,24);}
    else if(name=="undo"||name=="redo") {if(name=="redo"){p.translate(32,0);p.scale(-1,1);}p.setBrush(Qt::NoBrush);QPainterPath path;path.moveTo(7,12);path.cubicTo(26,5,31,26,14,27);p.drawPath(path);p.drawLine(7,12,8,4);p.drawLine(7,12,15,15);}
    else if(name=="save") {p.drawRoundedRect(QRectF(4,3,24,26),2,2);p.setBrush(blue);p.drawRect(QRectF(9,4,13,9));p.setBrush(Qt::white);p.drawRect(QRectF(9,18,14,10));}
    else if(name=="open") {p.setBrush(gold);p.setPen(QPen(QColor("#aa7a20"),1.5));p.drawPolygon(QPolygonF{{3,8},{13,8},{16,12},{28,12},{26,28},{3,28}});p.setBrush(QColor("#f5d284"));p.drawPolygon(QPolygonF{{3,28},{7,16},{30,16},{26,28}});}
    else if(name=="new") {p.drawRect(QRectF(6,3,20,26));p.setPen(QPen(teal,2));p.drawLine(16,10,16,23);p.drawLine(10,16,22,16);}
    else {p.setBrush(Qt::NoBrush);p.drawEllipse(QRectF(6,6,20,20));for(int i=0;i<8;++i){p.save();p.translate(16,16);p.rotate(i*45);p.drawLine(0,11,0,15);p.restore();}p.setBrush(teal);p.drawEllipse(QRectF(12,12,8,8));}
    QIcon result; result.addPixmap(pixels); return result;
}
