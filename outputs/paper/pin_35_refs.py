# -*- coding: utf-8 -*-
"""Pin manuscript to exactly 35 code-relevant references."""
from pathlib import Path

p = Path(__file__).resolve().parent / "ems_full_manuscript.md"
text = p.read_text(encoding="utf-8")

new_11 = r"""### 1.1 Cyclic irrigation and unobserved leftover water

In flat, low-lying paddy districts, cyclic irrigation collects drainage in a main canal and pumps it back into the supply (Takeda et al., 1997; Hama et al., 2010). Okura et al. (2025) monitored the Shiroyama–Jinbee area of the Imbanuma Land Improvement District (Chiba, Japan) in 2021 and 2022; subsurface lateral flow emerged as a large residual, and drought changed the cyclic reuse ratio. The package encodes those published identities in `cyclic/okura.py` and does not invent a new field campaign.

Plot-scale leftover \(L = IR + R - D - ET\) is not observed paddy by paddy. Bund seepage and preferential pathways motivate neighbour exchange in the leftover simplex (Walker and Rushton, 1986; Janssen et al., 2009; Tsubo et al., 2007), implemented as the hop graph and `head_equalise` teacher in `cyclic/agents.py` and `cyclic/lateral.py`. SWAT-Paddy shows that district models still struggle with rapid return flow (Tsuchiya et al., 2018). Irrigation “efficiency” language can mis-state reuse (Perry, 2007; Grafton et al., 2018), so the code scores capture (\(LFV_{\mathrm{pump}}/LfA\)) under known surface IR/D rather than claimed savings.

"""

new_12 = r"""### 1.2 Learning environments and open environmental software

The Gymnasium contract—spaces, `reset`/`step`, rewards and seeds (Sutton and Barto, 2018; Towers et al., 2024)—is implemented by `CyclicGym` (`cyclic/gym_env.py`) as `Cyclic-Imbanuma-v0` and `Cyclic-Imbanuma-Open-v0`. Closest prior gyms wrap crop or field irrigation models (Overweg et al., 2021; Gautron et al., 2022; Kelly and Foster, 2021; Kelly et al., 2024) or reservoir networks (Osika et al., 2026); none exposes thousands of paddies exchanging leftover water into a shared cyclic pool.

Simulation-based water control treats the simulator as fixed and searches policies on top (Giuliani et al., 2021). EMS open water software—Pywr (Tomlinson et al., 2020), Pywr-DRB (Hamilton et al., 2024), pytRIBS (Raming et al., 2025)—sets the packaging bar that `pip install -e ".[ems]"` and the three registered slots (`@register_forecaster`, `@register_ratio`, `@register_lateral`) aim to meet. Stable-Baselines3-style learners are an intended client API (Raffin et al., 2021).

"""

new_13 = r"""### 1.3 Purpose, contributions and stance

We follow EMS practice on purpose, evaluation and published workflows (Jakeman et al., 2006; Bennett et al., 2013; Hutton et al., 2016; Lemmen et al., 2024) and FAIR research software (Barker et al., 2022). Year-holdout routines in `cyclic/underground.py` and `cyclic/identify.py` follow the lesson that learned hydrology needs held-out seasons (Gauch et al., 2021). Equifinality (Beven, 2006) is explicit: many leftover fields can hit the same Table 5 pin. Hybrid process plus learning (Reichstein et al., 2019; Nearing et al., 2021) matches the code split—FAO-56 and Okura identities stay fixed in `cyclic/physics.py` / `cyclic/okura.py`; only leftover ratios and hop laws are learnable. Role-specific observation packs (`OBS_CHANNEL_PACKS` in `cyclic/agents.py`) make the problem partially observable by design (Kaelbling et al., 1998).

"""

new_rs = r"""### 2.3 Remote sensing

Satellite ET targets use NDVI-derived crop coefficients times reference ET (Allen et al., 1998; Er-Raki et al., 2007), coded in `cyclic/physics.py` (`kc_from_ndvi`) and `cyclic/satellite_et.py`. They score and observe; they do not overwrite simulated FAO-56 flux. `cyclic/copernicus.py` registers Copernicus Data Space credentials and searches Sentinel-2 over the district (Drusch et al., 2012). Weather forcing comes from Open-Meteo (Zippenfenig, 2023).

"""

new_71 = r"""### 7.1 What the environment makes possible

CyclicEnv turns Okura et al. (2025) into an executable object with three registered slots—farmer, ratio, hop—matching surface decisions, leftover partition and neighbour exchange in code. EMS evaluation norms (Jakeman et al., 2006; Bennett et al., 2013) and simulation-based control practice (Giuliani et al., 2021) motivate that split. Bund and toposequence work (Janssen et al., 2009; Tsubo et al., 2007) justify neighbour shares; Grafton et al. (2018) and Perry (2007) justify scoring capture rather than “savings.” Gymnasium (Towers et al., 2024), crop/basin gyms (Kelly et al., 2024; Osika et al., 2026) and EMS packages (Tomlinson et al., 2020; Hamilton et al., 2024; Raming et al., 2025) define the niche. Year holdouts (Gauch et al., 2021) and hybrid process–learning (Reichstein et al., 2019; Nearing et al., 2021) match the frozen-physics / learnable-leftover design.

"""

new_72 = r"""### 7.2 Lessons for benchmark design

- **Oracle leakage.** `et_match` sees observed block totals; it is package-internal.
- **Year instability.** Satellite farmer fidelity collapses from wet 2021 to dry 2022 (Gauch et al., 2021).
- **Strong null.** Zero discharge wins in `Cyclic-Imbanuma-v0`; beat that baseline, not random actions (Kelly et al., 2024).
- **Capacity versus transfer.** Per-paddy hop laws overfit `soil_flow_holdout`.
- **Equifinality bound.** A year-invariant capture field cannot exceed about 0.968 mean fidelity (Beven, 2006).
- **Fig. 8 versus IR_all.** Cyclic hard-copy matches Fig. 8; IR_all includes NCB (Okura et al., 2025).

"""

new_73 = r"""### 7.3 Limitations

One district, two seasons; digitised Fig. 8 and rescaled monthly drainage; hop teacher is a modelling assumption, not a measurement; east–west versus north–south neighbour shares are not identifiable; FAO-56 defaults are uncalibrated; pinned-IR throughput (about 6–7 steps s$^{-1}$) limits sample-hungry deep RL without vectorisation.

"""

new_74 = r"""### 7.4 Future work

Ingest original logger series; vectorise/batch the env; more seasons and districts; uncertainty on Table 4–5 pins; a frozen-API leaderboard for farmers, hop laws and simulation optimisation (Giuliani et al., 2021; Lemmen et al., 2024).

"""


def replace_between(src: str, start: str, end: str, new: str) -> str:
    i = src.index(start)
    j = src.index(end)
    return src[:i] + new + src[j:]


text = replace_between(text, "### 1.1 Cyclic irrigation", "### 1.2 Learning environments", new_11)
text = replace_between(text, "### 1.2 Learning environments", "### 1.3 Purpose", new_12)
text = replace_between(
    text, "### 1.3 Purpose", "The contributions of this article are:", new_13
)
text = replace_between(text, "### 2.3 Remote sensing", "## 3. Model", new_rs)
text = replace_between(text, "### 7.1 What the environment", "### 7.2 Lessons", new_71)
text = replace_between(text, "### 7.2 Lessons", "### 7.3 Limitations", new_72)
text = replace_between(text, "### 7.3 Limitations", "### 7.4 Future work", new_73)
text = replace_between(text, "### 7.4 Future work", "## 8. Conclusions", new_74)

text = text.replace(
    "Intended downstream users include Stable-Baselines3-style learners (Raffin et al., 2021) and simulation-optimisation stacks (Giuliani et al., 2021), without requiring either community to re-implement the district physics.",
    "Intended clients are Stable-Baselines3-style learners (Raffin et al., 2021) and simulation-optimisation stacks (Giuliani et al., 2021) against the frozen physics in `CyclicModel`.",
)
text = text.replace(
    "Intended clients are Stable-Baselines3-style learners (Raffin et al., 2021) and simulation-optimisation stacks (Giuliani et al., 2021), without requiring either community to re-implement the district physics.",
    "Intended clients are Stable-Baselines3-style learners (Raffin et al., 2021) and simulation-optimisation stacks (Giuliani et al., 2021) against the frozen physics in `CyclicModel`.",
)

refs = r"""## References

Allen, R.G., Pereira, L.S., Raes, D., Smith, M., 1998. Crop Evapotranspiration — Guidelines for Computing Crop Water Requirements. FAO Irrigation and Drainage Paper 56. FAO, Rome.

Barker, M., Chue Hong, N.P., Katz, D.S., Lamprecht, A.-L., Martinez-Ortiz, C., Psomopoulos, F., Harrow, J., Castro, L.J., Gruenpeter, M., Martinez, P.A., Honeyman, T., 2022. Introducing the FAIR Principles for research software. Scientific Data 9, 622. https://doi.org/10.1038/s41597-022-01710-x

Bennett, N.D., Croke, B.F.W., Guariso, G., Guillaume, J.H.A., Hamilton, S.H., Jakeman, A.J., Marsili-Libelli, S., Newham, L.T.H., Norton, J.P., Perrin, C., Pierce, S.A., Robson, B., Seppelt, R., Voinov, A.A., Fath, B.D., Andréassian, V., 2013. Characterising performance of environmental models. Environmental Modelling & Software 40, 1–20. https://doi.org/10.1016/j.envsoft.2012.09.011

Beven, K., 2006. A manifesto for the equifinality thesis. Journal of Hydrology 320, 18–36. https://doi.org/10.1016/j.jhydrol.2005.07.007

Drusch, M., Del Bello, U., Carlier, S., Colin, O., Fernandez, V., Gascon, F., Hoersch, B., Isola, C., Laberinti, P., Martimort, P., Meygret, A., Spoto, F., Sy, O., Marchese, F., Bargellini, P., 2012. Sentinel-2: ESA's optical high-resolution mission for GMES operational services. Remote Sensing of Environment 120, 25–36. https://doi.org/10.1016/j.rse.2011.11.026

Er-Raki, S., Chehbouni, A., Guemouria, N., Duchemin, B., Ezzahar, J., Hadria, R., 2007. Combining FAO-56 model and ground-based remote sensing to estimate water consumptions of wheat crops in a semi-arid region. Agricultural Water Management 87, 41–54. https://doi.org/10.1016/j.agwat.2006.02.004

Gauch, M., Mai, J., Lin, J., 2021. The proper care and feeding of CAMELS: How limited training data affects streamflow prediction. Environmental Modelling & Software 135, 104926. https://doi.org/10.1016/j.envsoft.2020.104926

Gautron, R., Padrón, E.J., Preux, P., Bigot, J., Maillard, O.-A., Emukpere, D., 2022. gym-DSSAT: a crop model turned into a reinforcement learning environment. arXiv:2207.03270.

Giuliani, M., Lamontagne, J.R., Reed, P.M., Castelletti, A., 2021. A state-of-the-art review of optimal reservoir control for managing conflicting demands in a changing world. Water Resources Research 57, e2021WR029927. https://doi.org/10.1029/2021WR029927

Grafton, R.Q., Williams, J., Perry, C.J., Molle, F., Ringler, C., Steduto, P., Udall, B., Wheeler, S.A., Wang, Y., Garrick, D., Allen, R.G., 2018. The paradox of irrigation efficiency. Science 361, 748–750. https://doi.org/10.1126/science.aat9314

Hama, T., Nakamura, K., Kawashima, S., 2010. Effectiveness of cyclic irrigation in reducing suspended solids load from a paddy-field district. Agricultural Water Management 97, 483–489. https://doi.org/10.1016/j.agwat.2009.11.009

Hamilton, A.L., Amestoy, T.J., Reed, P.M., 2024. Pywr-DRB: An open-source Python model for water availability and drought risk assessment in the Delaware River Basin. Environmental Modelling & Software 181, 106185. https://doi.org/10.1016/j.envsoft.2024.106185

Hutton, C., Wagener, T., Freer, J., Han, D., Duffy, C., Arheimer, B., 2016. Most computational hydrology is not reproducible, so is it really science? Water Resources Research 52, 7548–7555. https://doi.org/10.1002/2016WR019285

Jakeman, A.J., Letcher, R.A., Norton, J.P., 2006. Ten iterative steps in development and evaluation of environmental models. Environmental Modelling & Software 21, 602–614. https://doi.org/10.1016/j.envsoft.2006.01.004

Janssen, M., Lennartz, B., Wöhling, T., 2009. Water losses through paddy bunds: Methods, experimental data, and simulation studies. Journal of Hydrology 369, 142–153. https://doi.org/10.1016/j.jhydrol.2009.02.038

Kaelbling, L.P., Littman, M.L., Cassandra, A.R., 1998. Planning and acting in partially observable stochastic domains. Artificial Intelligence 101, 99–134. https://doi.org/10.1016/S0004-3702(98)00023-X

Kelly, T.D., Foster, T., 2021. AquaCrop-OSPy: Bridging the gap between research and practice in crop-water modeling. Agricultural Water Management 254, 106976. https://doi.org/10.1016/j.agwat.2021.106976

Kelly, T.D., Foster, T., Schultz, D.M., 2024. Assessing the value of deep reinforcement learning for irrigation scheduling. Smart Agricultural Technology 7, 100403. https://doi.org/10.1016/j.atech.2024.100403

Lemmen, C., Sommer, P.S., et al., 2024. Good modelling software practices. Ecological Modelling 498, 110890. https://doi.org/10.1016/j.ecolmodel.2024.110890

Nearing, G.S., Kratzert, F., Sampson, A.K., Pelissier, C.S., Klotz, D., Frame, J.M., Prieto, C., Gupta, H.V., 2021. What role does hydrological science play in the age of machine learning? Water Resources Research 57, e2020WR028091. https://doi.org/10.1029/2020WR028091

Okura, F., Kamei, A., Kato, T., 2025. Water balance analysis of hydrological processes in cyclic irrigation: A case study of the Imbanuma irrigation area in Chiba, Japan. Journal of Hydrology: Regional Studies 57, 102111. https://doi.org/10.1016/j.ejrh.2024.102111

Osika, Z., Rădulescu, R., Zatarain Salazar, J., Oliehoek, F.A., Murukannaiah, P.K., 2026. MORL4Water: A modular multi-objective reinforcement learning toolkit for water resource management. https://doi.org/10.65109/vsuw5215

Overweg, H., Berghuijs, H.N.C., Athanasiadis, I.N., 2021. CropGym: a reinforcement learning environment for crop management. arXiv:2104.04326.

Perry, C., 2007. Efficient irrigation; inefficient communication; flawed recommendations. Irrigation and Drainage 56, 367–378. https://doi.org/10.1002/ird.323

Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., Dormann, N., 2021. Stable-Baselines3: Reliable reinforcement learning implementations. Journal of Machine Learning Research 22 (268), 1–8.

Raming, L.W., Vivoni, E.R., Cederstrom, C.J., Hossain, M.A., Becerra, J.A., 2025. pytRIBS: An open, modular, and reproducible python-based framework for distributed hydrologic modeling. Environmental Modelling & Software 188, 106432. https://doi.org/10.1016/j.envsoft.2025.106432

Reichstein, M., Camps-Valls, G., Stevens, B., Jung, M., Denzler, J., Carvalhais, N., Prabhat, 2019. Deep learning and process understanding for data-driven Earth system science. Nature 566, 195–204. https://doi.org/10.1038/s41586-019-0912-1

Sutton, R.S., Barto, A.G., 2018. Reinforcement Learning: An Introduction, 2nd ed. MIT Press, Cambridge, MA.

Takeda, I., Fukushima, A., Tanaka, R., 1997. Non-point pollutant reduction in a paddy-field watershed using a circular irrigation system. Water Research 31, 2685–2692. https://doi.org/10.1016/S0043-1354(97)00117-6

Tomlinson, J.E., Arnott, J.H., Harou, J.J., 2020. A water resource simulator in Python. Environmental Modelling & Software 126, 104635. https://doi.org/10.1016/j.envsoft.2020.104635

Towers, M., Kwiatkowski, A., Terry, J., Balis, J.U., De Cola, G., Deleu, T., Goulão, M., Kallinteris, A., Krimmel, M., KG, A., Perez-Vicente, R., Pierré, A., Schulhoff, S., Tai, J.J., Tan, H., Younis, O.G., 2024. Gymnasium: A standard interface for reinforcement learning environments. arXiv:2407.17032.

Tsubo, M., Fukai, S., Tuong, T.P., Ouk, M., 2007. A water balance model for rainfed lowland rice fields emphasising lateral water movement within a toposequence. Ecological Modelling 204, 503–515. https://doi.org/10.1016/j.ecolmodel.2007.02.001

Tsuchiya, R., Kato, T., Jeong, J., Arnold, J.G., 2018. Development of SWAT-Paddy for simulating lowland paddy fields. Sustainability 10, 3246. https://doi.org/10.3390/su10093246

Walker, S.H., Rushton, K.R., 1986. Water losses through the bunds of irrigated rice fields interpreted through an analogue model. Agricultural Water Management 11, 59–73. https://doi.org/10.1016/0378-3774(86)90036-3

Zippenfenig, P., 2023. Open-Meteo.com Weather API. Zenodo. https://doi.org/10.5281/zenodo.7970649
"""

blocks = [b for b in refs.split("\n\n") if b.strip() and not b.startswith("##")]
assert len(blocks) == 35, len(blocks)

mapping = r"""## Appendix A. Reference–code map

| Code point | Reference |
|---|---|
| `cyclic/okura.py` tables, Eqs, Fig. 6/8 pins | Okura et al. (2025) |
| Cyclic irrigation practice | Takeda et al. (1997); Hama et al. (2010) |
| Bund seepage → hop graph / `head_equalise` | Walker and Rushton (1986); Janssen et al. (2009) |
| Lateral leftover on toposequence | Tsubo et al. (2007) |
| District paddy WB context | Tsuchiya et al. (2018) |
| Score capture not “efficiency savings” | Perry (2007); Grafton et al. (2018) |
| `Physics` FAO-56 dual \(K_c\) | Allen et al. (1998) |
| `kc_from_ndvi` / satellite ET target | Er-Raki et al. (2007) |
| `cyclic/copernicus.py` Sentinel-2 | Drusch et al. (2012) |
| Open-Meteo weather | Zippenfenig (2023) |
| `CyclicGym` / Gymnasium IDs | Towers et al. (2024); Sutton and Barto (2018) |
| Prior crop/irrigation gyms | Overweg et al. (2021); Gautron et al. (2022); Kelly and Foster (2021); Kelly et al. (2024) |
| Prior basin MORL toolkit | Osika et al. (2026) |
| EMS open water software bar | Tomlinson et al. (2020); Hamilton et al. (2024); Raming et al. (2025) |
| Policy search on simulators | Giuliani et al. (2021) |
| `@register_*` plugin slots / SB3 clients | Raffin et al. (2021) |
| `OBS_CHANNEL_PACKS` partial observability | Kaelbling et al. (1998) |
| Year-holdout (`underground_ratio_*`, `soil_flow_holdout`) | Gauch et al. (2021); Beven (2006) |
| Freeze FAO-56+Okura; learn leftover only | Reichstein et al. (2019); Nearing et al. (2021) |
| Software / evaluation stance | Jakeman et al. (2006); Bennett et al. (2013); Hutton et al. (2016); Barker et al. (2022); Lemmen et al. (2024) |

"""

start = text.index("## References")
end = text.index("## Appendix A. Reproducing the results")
rest = text[end:]
rest = rest.replace(
    "## Appendix A. Reproducing the results",
    "## Appendix B. Reproducing the results",
    1,
)
rest = rest.replace(
    "## Appendix B. Registering",
    "## Appendix C. Registering",
    1,
)
rest = rest.replace(
    "## Appendix C. Pre-submission",
    "## Appendix D. Pre-submission",
    1,
)

text = text[:start] + refs.strip() + "\n\n" + mapping + rest
p.write_text(text, encoding="utf-8")
print("refs", len(blocks))
print("wrote", p)
