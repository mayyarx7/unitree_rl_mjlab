# R1: LowState-Verbindung in MuJoCo

Lokal geprüft am 14.10.2026 unter WSL2, CycloneDDS 0.10.2 und MuJoCo 3.3.6.
Das Erfolgskriterium wurde mit den neu gebauten Originalprogrammen erreicht:
`r1_ctrl --network=lo --lowstate-only` meldet `Connected to robot.` und empfängt
fortlaufend Zustände. Es wurden keine Motorbefehle gesendet.

## Nachgewiesene Fehlerursache

1. **Große UDP-Datagramme erreichen den Empfänger nicht.** Ein separater
   Loopback-Test mit normalen Python-UDP-Sockets, ohne SDK oder DDS, ergab:

   | UDP-Nutzlast | sendto-Rückgabewert | Empfang |
   | --- | --- | --- |
   | 1000 Byte | 1000 | 1000 Byte |
   | 1400 Byte | 1400 | 1400 Byte |
   | 1472 Byte | 1472 | 1472 Byte |
   | 1473 Byte | 1473 | Timeout |
   | 2180 Byte | 2180 | Timeout |
   | 4000 Byte | 4000 | Timeout |

   Vor der Korrektur waren Publisher und Subscriber gematcht (`matched=1`,
   `incompatible_qos=0`), aber der Roh-Callback zählte zehn Sekunden lang null
   Nachrichten. Der SDK-Subscription-Wrapper blieb ebenfalls im Timeout.
   Discovery-Nachrichten kommen an; die größere LowState-Nutzlast fehlt.
   `strace` am Subscriber zeigte Empfangspuffer mit 65536 Byte und keinen
   erfolgreichen Empfang der großen LowState-Datagramme. Ein zu kleiner
   Subscriber-Puffer ist damit keine Erklärung.

2. **Die gesetzte XML-Datei wurde durch das SDK übergangen.** Bei
   `ChannelFactory::Init(domain, "lo")` erzeugt die lokal installierte statische
   SDK-Bibliothek eine eigene Interface-Konfiguration. Die XML-Einstellungen
   aus `CYCLONEDDS_URI` werden dabei nicht geladen. Mit leerem Interfaceargument
   wird die Umgebungs-Konfiguration geladen und das Tracing erzeugt Dateien.

Der eigentliche Bruch liegt im Transport großer UDP-Datagramme, nach erfolgreichem
DDS-Matching und vor Deserialisierung/Callback. Die ignorierte Konfiguration
verhinderte bisher Diagnose und eine wirksame Datagrammgrenzen-Korrektur.

## Kleine Korrektur

| Datei | Änderung |
| --- | --- |
| `simulate/src/main.cc` | Bei nichtleerem `CYCLONEDDS_URI` das Interfaceargument für SDK2 leer lassen. Ohne Umgebungs-Konfiguration gilt weiterhin das konfigurierte Interface. |
| `deploy/robots/r1/main.cpp` | Dieselbe Konfigurationsauswahl; zusätzlicher `--lowstate-only`-Modus mit dem bestehenden SDK-LowState-Subscriber, ohne LowCmd-Publisher und ohne FSM/Policy. Pro Sekunde wird der Tick angezeigt; bei unverändertem Tick erfolgt eine Meldung. |
| `deploy/include/param.h` | Optionale zusätzliche CLI-Optionen für den R1-Prüfmodus. Bestehende Aufrufer bleiben gültig. |
| `cyclonedds_local.xml` | `MaxMessageSize=1456 B`; `ParticipantIndex=auto` für Discovery über die Standard-Unicast-Ports; pro Prozess eigene Trace-Datei. |
| `simulate/tests/lowstate_probe.cc` | Passiver Zehn-Sekunden-Subscriber: zählt Nachrichten, zeigt Matching/QoS und prüft CRC, endliche Motor-/IMU-Werte und Quaternion-Norm. Rückgabewert 0 verlangt Empfang in mindestens fünf Intervallen und keine ungültigen Nachrichten. |
| `simulate/CMakeLists.txt` | Optionales Build-Target `lowstate_probe`, ausgeschlossen vom normalen Build. |

Bei gesetztem `CYCLONEDDS_URI` bestimmt die XML-Datei das Interface, auch wenn
`--network=lo` übergeben wird. Die DDS-Domain bleibt Simulator-Konfiguration bzw.
Domain 0 im R1-Controller. Beide müssen weiterhin dieselbe Domain verwenden.

`MaxMessageSize` begrenzt die UDP-Nutzlast. `FragmentSize` allein reicht nicht:
RTPS-Fragmente können in einem größeren UDP-Datagramm zusammengefasst werden.
Der unveränderte Standardwert `FragmentSize=1344 B` funktioniert zusammen mit
der neuen Begrenzung. Die empfangenen Traces zeigen LowState mit 2092 Byte
serialisierter Nutzlast in zwei `DATAFRAG`-Teilen, 1344 und 748 Byte.
Die größte beobachtete empfangene RTPS-UDP-Nutzlast im Abschlusstest war 1440 Byte.
Siehe [CycloneDDS-Konfiguration 0.10.2](https://github.com/eclipse-cyclonedds/cyclonedds/blob/0.10.2/src/core/ddsi/include/dds/ddsi/ddsi_cfgelems.h).

Ohne Multicast und mit adressbasierten Peers werden die Standard-Participant-Ports
abgefragt. `ParticipantIndex=auto` sorgt dafür, dass beide Prozesse auf solchen
Ports erreichbar sind. Mit dem zuvor effektiv ignorierten XML-Standardwert
`none` verwendeten sie zufällige Ports und fanden einander im isolierten Test
nicht. `${CYCLONEDDS_PID}` wird von der lokal geladenen Bibliothek unterstützt:
die finalen Dateien heißen beispielsweise `/tmp/cyclonedds-12760.log`.

## Lokale Implementierung und SDK-Vergleich

- Beide tatsächlichen Build-Konfigurationen verwenden `/opt/unitree_robotics/include`
  und dessen `ddscxx`-Header. Beide laden `/opt/unitree_robotics/lib/libddsc.so.0`
  und `libddscxx.so.0`, und linken `libunitree_sdk2.a` aus demselben Verzeichnis.
- Die HG-LowState-Header in `/opt`, `unitree_sdk2` und
  `unitree_mujoco/unitree_sdk2` haben identische SHA256-Hashes.
- Beide Endpunkte verwenden `rt/lowstate` und
  `unitree_hg::msg::dds_::LowState_` in Domain 0. Das tatsächliche Matching ist
  nachgewiesen, einschließlich `incompatible_qos=0`.
- 29 Aktuatoren führen in `main.cc` zur `G1Bridge`. Ihr LowState-Publisher
  initialisiert über `PublisherBase` den DDS-Writer vollständig.
- `trylock()` und `unlockAndPublish()` übergeben LowState an den
  `RealTimePublisher`-Thread. Dieser berechnet über den HG-Hook den CRC und ruft
  `publisher_->Write(outgoing, 0)` auf. Der Vorher-Test beweist, dass der Fehler
  nicht in FSM oder Policy liegt. Der erfolgreiche Empfang inklusive CRC-Prüfung
  nach der Transportkorrektur bestätigt den vollständigen Veröffentlichungspfad.
- Der LowState-Subscriber prüft im SDK DDS-Sample-Gültigkeit und aktualisiert
  dann den Empfangszeitpunkt; er verwirft LowState nicht aufgrund von
  `mode_machine`. Der unveränderte R1-Wert 0 ist der vom SDK akzeptierte
  Simulationsmodus. Eine Modusänderung ist für diesen Fehler nicht erforderlich.
- Der [offizielle Bridge-Code](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/simulate/src/unitree_sdk2_bridge.h)
  verwendet denselben Veröffentlichungsablauf. Am Bridge-Code war keine Änderung
  für die Verbindung erforderlich.

## Paketmeldung und verbleibende Unsicherheit

`UDP, bad length 2180 > 1472` bedeutet, dass die UDP-Längenangabe größer als die
vom Decoder verfügbare IP-Nutzlast ist. Der
[tcpdump-Decoder](https://github.com/the-tcpdump-group/tcpdump/blob/master/print-udp.c)
gibt diese Meldung bei diesem Vergleich aus, sofern er das Paket nicht als
fragmentiert behandelt. Sie allein beweist weder defekte DDS-Serialisierung
noch einen bestimmten WSL2-/Offloading-Fehler.

Der ursprüngliche Paketmitschnitt liegt nicht als PCAP vor. Ein neuer tcpdump-Test
scheiterte an fehlenden Capture-Rechten. Deshalb sind IP-Fragmentflags,
Capture-Länge und Offloading-Metadaten hier nicht vollständig ausgewertet.
Loopback meldet MTU 65536, trotzdem zeigt der unabhängige UDP-Test die Grenze
1472. Welche darunterliegende WSL2-, Windows-, Filter- oder Transportkomponente
die großen Datagramme verliert, ist nicht nachgewiesen. Der Fehler ist aber
unabhängig von DDS reproduziert, und die Korrektur umgeht ihn überprüfbar.

Für eine Untersuchung dieser unteren Schicht genügt der gleiche UDP-Grenztest
zusammen mit einem privilegierten Mitschnitt:
`sudo tcpdump -ni lo -s 0 -vv -w /tmp/udp-loopback.pcap udp`.
Dabei volle IP-/UDP-Längen, MF/Fragmentoffset und tatsächliche Capture-Längen
vergleichen. Eine Kernel- oder Offloading-Änderung ist für den DDS-Fix nicht nötig.

## Build und passive Wiederholung

Vom Projektverzeichnis:

```bash
cmake -S simulate -B simulate/build
cmake --build simulate/build --target unitree_mujoco lowstate_probe -j2
cmake --build deploy/robots/r1/build -j2
export CYCLONEDDS_URI="file://$PWD/cyclonedds_local.xml"
./simulate/build/unitree_mujoco
```

In einem zweiten Terminal, ebenfalls vom Projektverzeichnis:

```bash
export CYCLONEDDS_URI="file://$PWD/cyclonedds_local.xml"
./deploy/robots/r1/build/r1_ctrl --network=lo --lowstate-only
```

Optional in einem dritten Terminal mit derselben Umgebungsvariable:

```bash
./simulate/build/lowstate_probe
```

Abschlusstest mit echtem Simulator und beiden passiven Empfängern:

- Alle drei Targets erfolgreich kompiliert.
- Controller: `Connected to robot.`, anschließend Ticks 1614, 2612, 3614,
  4616, 5614, 6618, 7618; `mode_machine=0`.
- Probe: 6716 Nachrichten in zehn Sekunden inklusive Discovery-Anlauf;
  `invalid=0`, `matched=1`, `incompatible_qos=0`, Rückgabewert 0.
- Beide Prozesse wurden nach dem Test beendet; keine Steuerung aktiviert.
- Logs: `/tmp/r1-controller-final.log`, `/tmp/r1-probe-final.log`,
  `/tmp/r1-simulator-final.log`; DDS-Traces `/tmp/cyclonedds-12760.log`,
  `-12761.log`, `-12762.log`. Temporäre Logs können später entfernt werden.

## Mapping und Policy vor einer Steuerungsaktivierung

Die 24 `joint_ids_map`-Einträge ergeben exakt die Reihenfolge der 24 Gelenknamen
in den vorhandenen ONNX-Metadaten. Im geladenen `R1_C++.xml` passen alle drei
Sensorblöcke (q, dq, tau) zur Aktuatorreihenfolge. Nicht gesteuert werden
Motor 14 (waist_pitch), 20/21 (linkes wrist_pitch/yaw), 27/28 (rechtes wrist_pitch/yaw).
Das Policy-Format ist `obs=[1,83]`, `actions=[1,24]`; die konfigurierten
Beobachtungsgrößen ergeben 3+3+3+2+24+24+24 = 83.

**Die Policy-Konfiguration ist trotzdem noch nicht vollständig kompatibel:**
`params/deploy.yaml` setzt alle Aktionsskalierungen auf 0.25. Die ONNX-Metadaten
enthalten dagegen 0.150, 0.312, 0.375 und 0.412 je nach Gelenk; die aktuelle
Trainingskonfiguration verwendet ebenfalls `R1_ACTION_SCALE`. Vor Aktivierung
müssen diese Werte sowie Beobachtungssemantik, Reset-/Phasenverhalten und die
Behandlung der fünf zusätzlichen simulierten Gelenke abgeglichen werden.
Gelenkreihenfolge und Dimensionsgleichheit allein geben die Policy nicht frei.
Policy, Modell und bestehende Steuerungsparameter wurden für den DDS-Fix nicht
verändert.

Die bereits vor dieser Arbeit vorhandenen Änderungen an R1-CMake, Simulator-YAML,
Bridge-Debugausgabe und Roboter-XML bleiben bestehen. `git diff --check` meldet
zwei vorhandene Whitespace-Stellen im Roboter-XML; die hier geänderten Dateien
haben keine solchen Fehler.
