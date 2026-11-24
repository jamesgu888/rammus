# RAMMUS: Rotational Anomaly Monitoring in Motors Using Sound

James Gu and Tristan Zhu

## Can we hear a bearing fault?

Can we tell from a ten-second recording that a motor bearing is running off-center?
We'll compare two algorithms on the same recordings.
Both learn what is usual from healthy bearings.

1. **RMS amplitude detector (baseline).**
   This detector measures the overall sound level of the whole clip.
   It calculates one number, the root-mean-square (RMS) amplitude, and flags the clip if that number is unusually high or low for a healthy bearing.
2. **RAMMUS (our detector).**
   RAMMUS takes short FFTs and tracks the share of sound energy in each of five frequency bands.
   It flags a clip when one band's share stays unusually high or low, or repeatedly reaches an unusual level.
   For example, a bearing might develop a high-pitched whine while its overall sound level stays the same.
   The RMS detector could miss it, while RAMMUS could pick up the extra energy in the band containing the whine.

We expect RAMMUS to catch more faults at a similar false-alarm rate.
We'll test that prediction on recordings labeled healthy or faulty.

## Which recordings we'll use

Section 00 of MIMII DG [1] contains recordings of a shaft turning on two bearings, with healthy bearings and with bearings deliberately set off-center.
Each clip has one audio channel sampled at 16 kHz and includes background noise from a factory.
The filenames give us the shaft speed and tell us whether the bearing was healthy or faulty.

The 990 healthy training clips cover five speeds from 6,000 to 22,000 rpm, with 198 clips per speed.
We'll split whole clips at random within each speed and save the filenames and random seed.

| Dataset split | Experimental use | Healthy | Faulty |
| --- | --- | ---: | ---: |
| Reference training | Estimate feature means and standard deviations | 792 | 0 |
| Threshold calibration | Set each cutoff to the 95th percentile of healthy scores | 198 | 0 |
| Test at 5 training speeds | Measure detection and false-alarm rates | 50 | 50 |
| Test at 8 unseen speeds | Measure the same rates with fixed cutoffs | 50 | 50 |
| Unused target training set | Exclude from fitting and calibration | 10 | 0 |

The eight new speeds fall both within and beyond the training range.
We'll set aside their ten training clips and fix both alarm thresholds before we check the test labels.
This lets us see how often a change in speed triggers a false alarm.

## How we'll use Fourier analysis from EE102A

The recordings contain 16,000 samples per second.
From sampling in Chapter 6, we know the Nyquist frequency is 8 kHz, which sets the upper limit for our analysis.
We'll use Fourier analysis from Chapter 5 to see which frequencies carry the most power as the shaft turns.
In Python, we'll subtract the mean from each recording, take the first 1,024 samples, and compute their FFT.
That gives us the spectrum of a 64-millisecond segment, with frequency bins about 15.6 Hz apart.
We'll move forward 512 samples and repeat the calculation.
Comparing spectra every 32 milliseconds lets us see whether a tone lasts throughout the recording or appears in short bursts.

We'll divide each spectrum into five bands covering 0-0.5, 0.5-1, 1-2, 2-4 and 4-8 kHz.
For each band we'll add up its power and divide by the total power in that segment.
That tells us what share of the sound falls in each band, even if one recording is louder than another.
Across the ten seconds we'll keep the median share and the 95th percentile for each band.
The median can pick up a whine that lasts most of the clip.
The 95th percentile can pick up a band that gets louder in shorter, repeated bursts.

## When RAMMUS will flag a recording

Those five bands give us ten measurements per recording.
Using the 792 healthy clips, we'll find the average and standard deviation for each measurement.
For a new clip, we'll check how many standard deviations each value falls above or below its healthy average.
The biggest departure in either direction becomes the clip's score.
We'll leave out any measurement that has no variation in the reference clips.
We'll also keep track of the band that produced the score so we can see what set off an alarm.

For the RMS baseline, the score is the distance between the clip's RMS amplitude and the healthy average, measured in standard deviations.
For each detector we'll set the cutoff at the 95th percentile of its scores on the 198 healthy clips we held back.
A score above that cutoff will trigger an alarm.

## How we'll tell whether it worked

For each speed group, we'll count how many of the 50 faulty recordings each detector catches and how many of the 50 healthy recordings it flags by mistake.
At familiar speeds we'll also check how often a faulty clip gets a higher score than a healthy one.
The area under the ROC curve (AUROC) measures this across all possible alarm cutoffs.
To put a 95% confidence interval on the difference between the detectors, we'll repeatedly resample healthy and faulty clips separately, with replacement.
Both detectors will be compared on the same clips each time.

To see why RAMMUS raised an alarm, we'll mark the frequency band it flagged on the faulty bearing's spectrogram.
We'll compare that band with the same band in a healthy bearing's spectrogram at the same shaft speed, since speed itself changes the sound.

## References

[1] K. Dohi et al. *MIMII DG.* DCASE Workshop, 2022. Bearing recordings from Zenodo record 6529888.
