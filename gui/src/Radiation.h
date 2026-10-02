#pragma once
#include <QJsonObject>
#include <QVector>
#include <QString>
#include <limits>

struct RadiationMetrics {
    double radiatedPower=std::numeric_limits<double>::quiet_NaN();
    double incidentPower=std::numeric_limits<double>::quiet_NaN();
    double acceptedPower=std::numeric_limits<double>::quiet_NaN();
    QVector<double> directivity,gain,realizedGain;
    QString directivityUnavailable,gainUnavailable,realizedGainUnavailable;
};

// Dimensionless 2D circular metrics. Raw DFT powers share the same W s^2 scale.
RadiationMetrics radiationMetrics(const QJsonObject& results,int runIndex,int frequencyIndex);
