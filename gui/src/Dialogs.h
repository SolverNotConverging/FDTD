#pragma once
#include <QDialog>
#include <QJsonObject>
#include <QMap>
class QFormLayout;
class QComboBox;
class QLineEdit;
class QPlainTextEdit;
class QDoubleSpinBox;
class Project;

class ObjectDialog : public QDialog {
    Q_OBJECT
public:
    ObjectDialog(Project* project,const QString& category,QJsonObject value,int index=-1,QWidget* parent=nullptr);
    QJsonObject value() const { return result_; }
private:
    Project* project_; QString category_; QJsonObject initial_,result_; int index_;
    QLineEdit* name_; QComboBox* kind_; QComboBox* material_=nullptr;
    QFormLayout* form_; QWidget* fields_; QPlainTextEdit* vertices_=nullptr;
    QLineEdit* frequencies_=nullptr;
    QMap<QString,QDoubleSpinBox*> numbers_; QMap<QString,QComboBox*> choices_;
    void rebuild(); void finish();
    void number(const QString& key,const QString& title,double fallback,double minimum=-1e6,double maximum=1e6,int decimals=6);
    void choice(const QString& key,const QString& title,const QStringList& values,const QString& fallback);
    double n(const QString& key) const;
    QString c(const QString& key) const;
};

class SettingsDialog : public QDialog {
    Q_OBJECT
public:
    explicit SettingsDialog(const QJsonObject& project,QWidget* parent=nullptr);
    QJsonObject value() const { return result_; }
private:
    QJsonObject result_;
    QMap<QString,QDoubleSpinBox*> numbers_;
    QComboBox* polarization_; QComboBox* study_;
    void finish();
};
