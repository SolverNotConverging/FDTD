#include "Radiation.h"
#include <QJsonArray>
#include <QSet>
#include <cmath>
#include <algorithm>

namespace {
constexpr double pi=3.14159265358979323846;
constexpr double invalid=std::numeric_limits<double>::quiet_NaN();
double number(const QJsonValue& value) { return value.isDouble()?value.toDouble():invalid; }
QJsonArray row(const QJsonObject& run,const QString& key,int frequency) {
    const auto array=run[key].toArray(); return frequency>=0&&frequency<array.size()?array[frequency].toArray():QJsonArray();
}
bool transportsPower(const QJsonObject& results,const QJsonValue& channel,int frequency) {
    const auto pair=channel.toArray();
    for(const auto& item:results["mesh"].toObject()["port_modes"].toArray()) {
        const auto port=item.toObject(); if(port["name"].toString()!=pair[0].toString()) continue;
        for(const auto& itemMode:port["modes"].toArray()) {
            const auto mode=itemMode.toObject(); if(mode["index"].toInt()!=pair[1].toInt()) continue;
            const auto valid=mode["valid"].toArray();
            if(frequency>=0&&frequency<valid.size()&&valid[frequency].isBool()) return valid[frequency].toBool();
        }
    }
    return true; // Lumped terminals and older results without modal metadata.
}
double wavePower(const QJsonObject& results,const QJsonObject& run,const QString& prefix,int frequency,const QVector<int>& indices) {
    const auto real=row(run,prefix+"_real",frequency),imag=row(run,prefix+"_imag",frequency);
    const auto channels=results["channels"].toArray();
    double sum=0;
    for(const int i:indices) {
        if(!transportsPower(results,channels[i],frequency)) continue;
        if(i>=real.size()||i>=imag.size()) return invalid;
        const double r=number(real[i]),j=number(imag[i]); if(!std::isfinite(r)||!std::isfinite(j)) return invalid;
        sum+=r*r+j*j;
    }
    return sum;
}
}

RadiationMetrics radiationMetrics(const QJsonObject& results,int runIndex,int frequencyIndex) {
    RadiationMetrics metrics; const auto runs=results["runs"].toArray(),angles=results["angles_deg"].toArray();
    if(runIndex<0||runIndex>=runs.size()) return metrics;
    const auto run=runs[runIndex].toObject(); const auto powers=row(run,"power",frequencyIndex);
    metrics.directivity.fill(invalid,angles.size()); metrics.gain=metrics.realizedGain=metrics.directivity;
    bool fullCircle=powers.size()==angles.size()&&angles.size()>1;
    double integral=0;
    for(int i=0;fullCircle&&i<angles.size();++i) {
        const double angle=number(angles[i]),power=number(powers[i]);
        if(!std::isfinite(angle)||!std::isfinite(power)||power<0) { fullCircle=false; break; }
        if(i) {
            const double delta=angle-number(angles[i-1]); if(delta<=0) {fullCircle=false;break;}
            integral+=.5*(number(powers[i-1])+power)*delta*pi/180.;
        }
    }
    fullCircle=fullCircle&&std::abs(number(angles.last())-number(angles.first())-360.)<1e-6;
    if(fullCircle&&std::isfinite(integral)&&integral>0) {
        metrics.radiatedPower=integral;
        for(int i=0;i<angles.size();++i) metrics.directivity[i]=2*pi*number(powers[i])/integral;
    } else metrics.directivityUnavailable="Directivity requires a complete 360-degree pattern with nonzero radiated power.";

    const auto channels=results["channels"].toArray(),driven=run["driven_channels"].toArray(); QSet<QString> feedNames;
    bool portDriven=!driven.isEmpty();
    for(const auto& drive:driven) {
        if(!channels.contains(drive)) {portDriven=false;break;}
        feedNames.insert(drive.toArray()[0].toString());
    }
    if(!portDriven) {
        metrics.gainUnavailable=metrics.realizedGainUnavailable="Gain requires port-only excitation with measured incident and reflected feed power.";
        return metrics;
    }
    for(const auto& drive:driven) if(!transportsPower(results,drive,frequencyIndex)) {
        metrics.gainUnavailable=metrics.realizedGainUnavailable="Gain is unavailable where a driven waveguide mode does not propagate.";
        return metrics;
    }
    // Every mode at a driven physical port contributes to reflected feed power.
    // Receiving ports remain loads, so their outgoing power is not feed reflection.
    QVector<int> feeds;
    for(int i=0;i<channels.size();++i) if(feedNames.contains(channels[i].toArray()[0].toString())) feeds.append(i);
    metrics.incidentPower=wavePower(results,run,"incoming",frequencyIndex,feeds);
    const double reflected=wavePower(results,run,"outgoing",frequencyIndex,feeds);
    metrics.acceptedPower=metrics.incidentPower-reflected;
    double peakIncident=0;
    for(int f=0;f<results["frequencies_ghz"].toArray().size();++f) {
        const double value=wavePower(results,run,"incoming",f,feeds); if(std::isfinite(value)) peakIncident=std::max(peakIncident,value);
    }
    const bool incidentValid=std::isfinite(metrics.incidentPower)&&metrics.incidentPower>0&&metrics.incidentPower>peakIncident*1e-12;
    const bool acceptedValid=incidentValid&&std::isfinite(metrics.acceptedPower)&&metrics.acceptedPower>metrics.incidentPower*1e-12;
    if(!fullCircle) metrics.gainUnavailable=metrics.realizedGainUnavailable="Gain requires a complete, finite 360-degree radiation pattern.";
    else {
        if(incidentValid) for(int i=0;i<angles.size();++i) metrics.realizedGain[i]=2*pi*number(powers[i])/metrics.incidentPower;
        else metrics.realizedGainUnavailable="Realized gain is unavailable where the incident feed spectrum is zero, invalid or too weak.";
        if(acceptedValid) for(int i=0;i<angles.size();++i) metrics.gain[i]=2*pi*number(powers[i])/metrics.acceptedPower;
        else metrics.gainUnavailable="Gain is unavailable where accepted feed power is nonpositive, invalid or too weak.";
    }
    return metrics;
}
