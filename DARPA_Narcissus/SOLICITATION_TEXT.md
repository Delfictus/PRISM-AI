# DARPA Narcissus - Original Solicitation Text

**Topic Number:** HR0011SB20254-14
**Topic Title:** Narcissus
**Technology Areas:** Sensors
**Modernization Priorities:** Advanced Computing and Software | Space Technology

## Keywords
Situational awareness, telescope, corrective adaptive optics, faint object detection, machine learning, neural networks, novel optics, relay optics

---

## OBJECTIVE

In a previous research effort titled WITH-US (Window-glass Telescope for Highly-Compensated Ubiquitous Sensing), DARPA validated that it may be possible to design a large collecting area telescope system, where the main light-collecting optical surface is window(s) already installed on a commercial office building. Millions of square meters of commercial float glass on skyscrapers and other buildings, if usable as a primary optic, could allow for a rapid proliferation of upward-looking imagers for an order-of-magnitude jump in transient target awareness.

WITH-US largely focused on the design of hardware to focus collected light from a subset of glass on a skyscraper into secondary optics. This topic, Narcissus, seeks to develop the computational imaging capability to sense-make from this new imaging system, and create performance estimates in the presence of noise sources such as skyglow, temperature variation, wind, and atmospheric turbulence, among others.

A key focus is on low-mass scalability (i.e., a software over hardware approach). In other words, a crucial aspect of this large collection aperture is that not all light from every windowpane will be able to be coherently combined into the same secondary aperture. Narcissus aims to rapidly develop computational imaging, artificial intelligence, or machine learning innovations that can perform the coherent and incoherent combination of incident photon flux to enable compensated image reconstruction for light reflected from every window of a skyscraper. The eventual end goal is to utilize the massive quantity of in-situ commercial building windows as a tool for sensing faint objects in space or the upper atmosphere.

---

## DESCRIPTION

Commercial high-rise building window surfaces tend to be transparent and flat; small-scale surface imperfections and waviness degrade imaging quality through large sections of the window. However, other attributes make them intriguing for use as a possible light-collecting mirror. Most notably, their ubiquitous nature and extremely cheap cost at scale allow them to concentrate large amounts of light in such a way that very dim objects could be detected. A slight concave shape is common; this provides magnification key to faint object detection. Commercial float glass has an inherently ultra-smooth surface ideal for reflecting optical signals and is highly stable (excepting windy conditions). It is also possible to use things like bounced-collection of edges of neighboring buildings for metrology corrections, and common low-emissivity coatings for infrared functionality.

The use of such windows as ad-hoc telescope mirrors provides a dramatic boost in light-gathering ability. This effort will expand this proof of concept to the development of computational/lensless imaging methods for "the building scale", i.e., the performance of a whole series of windows on a skyscraper.

### Target System Specifications

Specifically, Narcissus will analyze a building that is:
- **Height:** 10 stories tall
- **Dimensions:** 30 meters x 30 meters
- **Total imaging area:** 900 square meters
- **Fill factor:** 100% (all glass)
- **Window pane size:** 1 meter x 1 meter
- **Total panes:** 900 panes

This is a building qualitatively similar to the upper part of "The Edge", 30 Hudson Yards, New York City.

### Challenge

It may be expected that a single secondary aperture may be able to collect light from a 3 x 3 grid of window panes (coherent light combination). However, this results in 100 secondary apertures required to collect all the light from the skyscraper, defeating the low-complexity and low-cost motivations behind using skyscrapers as telescopes.

Therefore, Narcissus is interested in methods to incoherently and coherently combine light collected from the surface of an entire skyscraper, to create detections and object tracking inherent to aircraft (during the day) and spacecraft (during the night) detections.

### Technical Approach

Specifically, Narcissus seeks methods of image compensation particular to this unconventional system to invert the encoded spatial, spectral, and even temporal optical information from the digital sensor and maximize the performance of window-glass systems. Even in a "bumpy" window glass case, this distorted reflector still deterministically encodes the light flux distribution from the scene onto the detector pixels.

In general, methods of applying numerical processing to sensor data, as well as machine learning algorithms, are expected to provide substantial improvements to overall sensing capability. Developed algorithms may also be self-calibrating, i.e., even if the window glass mirror slowly changes its shape and distortion, the algorithm would be able to recalibrate or retrain to preserve performance.

---

## PHASE I (9 months)

Phase I consists of a base period of nine (9) months to design new computational or lensless imaging approaches that can specifically ingest light from 900 contiguous panes of 1 x 1-meter commercial float glass, combine them, and create simulations that can measure the performance of such a large system. A successful Phase I effort will involve modeling and simulation to achieve this goal, specific to the 900-pane building described in this solicitation.

### Milestones

**Month 1 (within first 30 days):** Virtual kickoff meeting. Presentation of computational imaging/machine learning/other technical approach to the problem.

**Month 3: PI Meeting #1**
- Initial demonstration of a general caustic image simulator that simulates detector images from arbitrary "focal planes" and with arbitrary scenes, across a variety of commonly anticipated window shape configurations (e.g., surface concavity, undulation, reflectivity, angle of incidence/reflectance, etc.).
- Baseline for a low-complexity and scalable hardware solution that coherently combines light from multiple panes, and initial algorithm development for incoherent image analysis.
- To maximize collaboration toward the goals of the program, proposers should anticipate presentation to an audience of other performers (non-proprietary), with a proprietary follow-up to a government-only audience.

**Month 6:** Interim project update. Government-only audience.

**Month 9: Phase I Final project update**
- Government-only audience
- Demonstration of "best results" toward the problem, with description of simulation tools and specific algorithms used for data inversion.
- Present updates to baseline hardware solution, if applicable.
- Scalability study that estimates cost of a combined hardware-software system to use a single building as an imager.
- To maximize collaboration toward the goals of the program, proposers should anticipate presentation to an audience of other performers (non-proprietary), with a proprietary follow-up to a government-only audience.

---

## PHASE II (12 months)

To show sufficient progress to proceed to a Phase II, which has a base period of twelve (12) months, proposers should show:
1. A well-developed and scalable caustic image simulator that simulates detector images from arbitrary "focal planes", specific to the 900-pane building described in this solicitation
2. A computational imaging approach to perform both coherent and incoherent light combination and analysis

Phase II will further develop modeling methods and advance a cohesive optical design to the Critical Design Review (CDR) level for a fieldable system at a prototype level, capable of performing astronomical measurements of known dim objects. Phase II will validate the capabilities determined during Phase I through small-scale hardware testing. Based on simulated results, small-scale window characterization testing, and development of a CDR-level design, Phase II efforts should project the notional ability for a larger system to accomplish sufficient resolution for faint object detection at realistic standoff measurement distances typical of an urban environment.

### Milestones

**Month 15:** Preliminary Design Review (PDR) level design encompassing hardware and software algorithms.

**Month 18:** PI Meeting #3.

**Month 21:** Small-scale laboratory demonstration of the window characterization/measurement software, its optical correction subsystem, its incorporation with a physical installed window, and resulting image compensation results fed into the larger building-level (900 pane) simulator. Present performance estimations of building-level sensor performance under relevant varying conditions (wind, temperature, concavity, etc.).

**Month 24:** Critical Design Review (CDR) level design, with final laboratory and software demonstrations. Present results determining the feasibility for the integration of components into a fieldable system at a prototype level, capable of performing measurements of known dim objects (e.g., stars).

---

## PHASE III DUAL USE APPLICATIONS

Using ordinary mm-scale thickness window glass, in combination with numerical algorithms, to generate accurate images of distant (telescopic) scenery, will be a paradigm shift for numerous imaging applications.

### Military Applications

**Ground-to-space:**
- Ground-based Space Domain Awareness (SDA)
- Satellite surveillance

**Air-to-air:**
- Advanced targeting pod sensors for low-cost Unmanned Aerial Systems

### Commercial Applications

Due to the intrinsically inexpensive nature of this concept (only requiring GPU-scale processing and simple, commercially available float-glass), this development is expected to be broadly commercializable.

**Ground-based telescopes:**
- Thin (mm-scale) glass mirrors
- Much lower mass platforms (by an order of magnitude or more)
- New ways to scale to inexpensive, Very Large Telescope (VLT) systems

**Problem domains:**
- Distant (GEO, cislunar) satellite tracking from Earth
- Commercial astronomy

**Space Situational Awareness (SSA) and Space Domain Awareness (SDA):**
- Ground-based imaging systems will directly benefit
- Proliferation of SSA/SDA imaging to cities across the world
- Potential for cost- and size-scalable Very Large Telescope (VLT) systems
- Plays directly into existing architecture goals specified by USSTRATCOM for Space Domain Awareness

**Commercial and civil applications:**
- Widespread commercial astronomy applications
- Civil space monitoring

---

## REFERENCES

1. V. Boominathan, J. T. Robinson, L. Waller, and A. Veeraraghavan, "Recent advances in lensless imaging," Optica 9, pp. 1–16, Jan 2022.

2. S. Nayar, "Computational cameras redefining the image," Computer 39, pp. 30–38, Jan 2006.

---

## TECHNICAL POINT OF CONTACT (TPOC)
None

---

*Solicitation captured: September 30, 2025*
*For internal use by Neuromorphic-Quantum Platform team*
