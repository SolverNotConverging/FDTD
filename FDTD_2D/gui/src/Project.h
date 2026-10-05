#pragma once
#include <QObject>
#include <QJsonObject>
#include <QJsonArray>
#include <QUndoStack>

class Project : public QObject {
    Q_OBJECT
public:
    explicit Project(QObject* parent=nullptr);
    QJsonObject data() const { return data_; }
    QJsonArray items(const QString& category) const { return data_.value(category).toArray(); }
    QJsonObject item(const QString& category,int index) const;
    QJsonObject settings() const { return data_.value("settings").toObject(); }
    QUndoStack* undoStack() { return &undo_; }
    bool dirty() const { return !undo_.isClean(); }
    void reset(const QJsonObject& data);
    void replace(const QJsonObject& data,const QString& label);
    void add(const QString& category,const QJsonObject& item);
    void edit(const QString& category,int index,const QJsonObject& item);
    void remove(const QString& category,int index);
    void move(const QString& category,int index,double dx,double dy);
    QString uniqueName(const QString& prefix) const;
    bool nameExists(const QString& name,const QString& category={},int except=-1) const;
    bool load(const QString& path,QString* error);
    bool save(const QString& path,QString* error);
    static QJsonObject empty();
signals:
    void changed();
private:
    friend class ProjectEdit;
    QJsonObject data_;
    QUndoStack undo_;
    void apply(const QJsonObject& value);
};
