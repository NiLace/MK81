# Changelog

All notable changes to **MK 81**, newest first.

Versions follow the LV2 rules: the plugin's URI is the major version, so a release is numbered
`1.minor.micro`; a release has a minor number above zero, and even minor and micro numbers.

---

## 1.4.0 — 11-Nov-2026

- **A new engine: the whole amplifier is now one circuit, solved sample by sample.** The preamp, the phase inverter,
  the push-pull power stage with its output transformer and feedback, the power supply with its rectifier and sag, and
  the load of a real 4x12 cabinet are computed together from the schematic, at 2x the sample rate. The earlier
  power stage (a small network trained against the circuit) is gone; the `engine` port stays so older sessions load,
  and is ignored.
- **HQ**: the letters under the badge switch the amp to 4x oversampling, a little closer to the circuit on the loudest
  playing, for about 1.6 to 1.8 times the CPU. Off (2x) is the default.
- **A fifth version, the 100 W reissue**, with its built-in effects loop card: the loop is now that card, in series,
  with a **LEVEL** switch (PEDAL for stomp boxes, the default; LINE for rack effects) in place of the FX Mix knob.
  `fx_mix` stays as a port for older sessions.
- The Era switch acts on the 100 W originals only (the 50 W head and the reissue have one supply); its lever sits in
  the middle on those.
- Latency: **85 samples at every sample rate** (it was 47 at 48 kHz in 1.2.2), the same at 2x and 4x, with the loop on
  or off, so no switch moves it.
- Switching the model while the effects loop is on is clean, and a host stopping and restarting the transport with the
  loop on restarts it as a freshly loaded plugin would.
- Editor: the HQ letters and the LEVEL switch; it no longer crashes a host that destroys the editor's window first,
  ignores a non-finite value from the host instead of keeping it, opens in about a quarter of the time, can be
  embedded in a 32-bit (compositing) host window, and closing it can no longer crash the host later.
- The plugin reads nothing from the environment.
- **Very hot input is safe.** Fuzz pedals at full volume in front of the amp, or loud broadband noise, could make the
  circuit solver accept a wrong answer: the amp then burst out far louder than full scale and stalled the host for a
  moment. It no longer happens, and the solver's work on any single sample now has a ceiling, so no input can stall the
  audio thread.
- **The same sound at every buffer size**: moving a knob, flipping the Era switch or HQ right after the plugin starts or
  the transport restarts gives the same output whatever block size the host runs, down to one sample.

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
