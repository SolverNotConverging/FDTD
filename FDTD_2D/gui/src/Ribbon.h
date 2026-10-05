#pragma once
#include <QWidget>
#include <QIcon>
class QAction;
class QHBoxLayout;
class QTabBar;
class QStackedWidget;
class QToolButton;

// A Qt Widgets ribbon: every button shares the original menu QAction.
class RibbonGroup : public QWidget {
public:
    explicit RibbonGroup(const QString& title, QWidget* parent=nullptr);
    QToolButton* addCommand(QAction* action, const QString& label, const QString& icon);
    void addControl(QWidget* control);
private:
    QHBoxLayout* commands_;
};

class Ribbon : public QWidget {
public:
    explicit Ribbon(QWidget* parent=nullptr);
    int addPage(const QString& title);
    RibbonGroup* addGroup(int page, const QString& title);
    void setCurrentPage(int page);
    int currentPage() const;
    static QIcon icon(const QString& name);
private:
    QTabBar* tabs_;
    QStackedWidget* pages_;
};
