# Changelog

All notable changes to **MK 81**, newest first.

Versions follow the LV2 rules: the plugin's URI is the major version, so a release is numbered
`1.minor.micro`; a release has a minor number above zero, and even minor and micro numbers.

---

## 1.2.2

- The Bias readout on the back panel now shows the right idle current for the 6550 versions: 25, 40 and 55 mA per
  valve from cold through factory to hot, as on the EL34 versions. 1.2.0 printed 42-92 mA there. The sound does not
  change: only the number the panel shows.

## 1.2.0 — first public release

- A British valve amplifier head of the early 1980s, modelled from its circuit, in four versions: 100 W or
  50 W, with EL34 or 6550 power valves. The preamp is solved sample by sample; the power stage, the supply's
  sag and the speaker load are a small recurrent network per version.
- Front: Presence, Bass, Middle, Treble, Master, Pre-Amp and Output, the HIGH and LOW inputs, Power and the
  model display. Back: Full, Preamp and Power bypass, Analog Noise, Era, Bias, Cabinet Feel and an effects
  loop with its own send and return.
- A rendered interface: click the grille cloth to turn the amp round, rest the pointer on a
  control to see what it does, and read a knob's value in place of its name while you turn it.
- Latency 47 samples at 48 kHz, reported to the host. Generic x86-64: no AVX2 required.
