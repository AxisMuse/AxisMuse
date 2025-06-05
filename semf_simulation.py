# A simple simulation of a Synthetic Entangled Mesh Field (SEMF)
# This script implements the SEMFNode and SEMFLattice classes and runs
# a short animation using matplotlib. The implementation is derived from
# the partial code provided in the conversation.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from scipy.special import erfc
from scipy.fft import fft, ifftn, fftn
import random


class SEMFNode:
    """Single node in the synthetic entangled mesh."""

    def __init__(self, position, initial_phase, vibrational_modes=None, alpha=1.0):
        self.position = np.array(position, dtype=float)
        self.phase = float(initial_phase)
        self.entropy = 0.0
        self.entangled_nodes = []
        self.casimir_pressure = 0.0
        self.real_field = np.zeros(3)
        self.reciprocal_field = np.zeros(3)
        self.temporal_phase_profile = [self.phase]
        self.recursive_anchor_strength = 1.0
        self.vibrational_modes = vibrational_modes or []
        self.is_resonant = False
        self.is_collapse_center = False
        self.alpha = alpha
        self.glyph_stability = 0.0

    def sample_field(self, field_vector):
        self.real_field = field_vector

    def update_phase(self, delta_phase):
        self.phase += delta_phase
        self.temporal_phase_profile.append(self.phase)
        if len(self.temporal_phase_profile) > 10:
            self.temporal_phase_profile.pop(0)

    def compute_entropy(self):
        field_norm = np.linalg.norm(self.real_field)
        phase_var = np.var(self.temporal_phase_profile) if len(self.temporal_phase_profile) > 1 else 0.0
        self.entropy = field_norm ** 2 + phase_var
        self.is_collapse_center = self.entropy < 0.2

    def detect_resonance(self, external_vibrations, threshold=0.1):
        self.is_resonant = any(
            abs(ev - vm) < threshold for ev in external_vibrations for vm in self.vibrational_modes
        )
        if self.is_resonant:
            self.recursive_anchor_strength *= 1.05
        else:
            self.recursive_anchor_strength *= 0.99

    def calculate_casimir_pressure(self, other_node, distance):
        if distance < 1e-4:
            self.casimir_pressure = 100.0
        else:
            self.casimir_pressure = -(np.pi ** 2) / (240 * distance ** 4)
        return self.casimir_pressure

    def apply_resonant_ewald_field(self, field_value, alpha=1.0):
        distance = np.linalg.norm(field_value)
        if distance < 1e-4:
            return np.zeros(3)
        real_component = erfc(np.sqrt(alpha) * distance) / distance
        self.real_field *= real_component
        return self.real_field

    def apply_temporal_ewald_coupling(self):
        if len(self.temporal_phase_profile) > 3:
            phases = np.array(self.temporal_phase_profile)
            spectrum = fft(phases)
            dominant = np.argmax(np.abs(spectrum[1:len(spectrum)//2])) + 1
            if dominant > 0:
                corr = 0.1 * np.sin(2 * np.pi * dominant / len(phases))
                self.phase += corr

    def form_glyph(self):
        if not self.is_collapse_center:
            return False
        self.glyph_stability = min(1.0, self.glyph_stability + 0.1)
        return self.glyph_stability > 0.5


class SEMFLattice:
    """Lattice of SEMF nodes."""

    def __init__(self, dimensions=(6, 6, 6), spacing=1.0):
        self.dimensions = dimensions
        self.spacing = spacing
        self.nodes = []
        self.simulation_step = 0
        self.alpha = 1.0
        self.external_vibrations = np.random.rand(5) * 2.0
        self.entanglement_threshold = 2.0 * spacing
        self.collapse_zones = []
        self.glyphs = []
        self.time_slices = []
        self._initialize_lattice()

    def _initialize_lattice(self):
        for x in range(self.dimensions[0]):
            for y in range(self.dimensions[1]):
                for z in range(self.dimensions[2]):
                    pos = [
                        x * self.spacing + np.random.normal(0, 0.1),
                        y * self.spacing + np.random.normal(0, 0.1),
                        z * self.spacing + np.random.normal(0, 0.1),
                    ]
                    init_phase = np.random.rand() * 2 * np.pi
                    modes = np.random.rand(np.random.randint(3, 6)) * 2.0
                    node = SEMFNode(pos, init_phase, modes, self.alpha)
                    self.nodes.append(node)
        self._establish_entanglement()

    def _establish_entanglement(self):
        for i, node in enumerate(self.nodes):
            for j, other in enumerate(self.nodes):
                if i == j:
                    continue
                dist = np.linalg.norm(node.position - other.position)
                if dist <= self.entanglement_threshold:
                    node.entangled_nodes.append(j)
                    node.calculate_casimir_pressure(other, dist)

    def simulate_step(self):
        self.simulation_step += 1
        if self.simulation_step % 10 == 0:
            self.external_vibrations = np.random.rand(5) * 2.0
        new_collapse = []
        for idx, node in enumerate(self.nodes):
            local_field = self._sample_local_field(idx)
            node.sample_field(local_field)
            node.apply_resonant_ewald_field(local_field, self.alpha)
            node.detect_resonance(self.external_vibrations)
            node.apply_temporal_ewald_coupling()
            delta = 0.1 * np.sin(np.sum(node.real_field))
            if node.is_resonant:
                delta *= 2.0
            node.update_phase(delta)
            node.compute_entropy()
            if node.is_collapse_center:
                new_collapse.append(idx)
                if node.form_glyph() and idx not in self.glyphs:
                    self.glyphs.append(idx)
        self._apply_casimir_stabilization()
        self._update_positions()
        self.collapse_zones = new_collapse
        self._store_time_slice()

    def _sample_local_field(self, idx):
        node = self.nodes[idx]
        local = np.zeros(3)
        for j in node.entangled_nodes:
            other = self.nodes[j]
            direction = other.position - node.position
            distance = np.linalg.norm(direction)
            if distance > 1e-4:
                direction /= distance
                phase_diff = other.phase - node.phase
                strength = np.sin(phase_diff) / (distance ** 2)
                local += direction * strength
        local += np.random.normal(0, 0.05, 3)
        return local

    def _apply_casimir_stabilization(self):
        for i, node in enumerate(self.nodes):
            force = np.zeros(3)
            for j in node.entangled_nodes:
                other = self.nodes[j]
                direction = other.position - node.position
                distance = np.linalg.norm(direction)
                if distance > 1e-4:
                    direction /= distance
                    pressure = node.calculate_casimir_pressure(other, distance)
                    force += direction * pressure
            node.casimir_pressure = np.linalg.norm(force)
            node.real_field += force * 0.01

    def _update_positions(self):
        for node in self.nodes:
            delta = np.zeros(3)
            for j in node.entangled_nodes:
                other = self.nodes[j]
                direction = other.position - node.position
                distance = np.linalg.norm(direction)
                if distance > 1e-4:
                    direction /= distance
                    optimal = self.spacing
                    attraction = 0.01 * (distance - optimal)
                    net_force = attraction - 0.005 * node.casimir_pressure
                    delta += direction * net_force
            if node.is_resonant:
                delta *= 0.5
            if node.is_collapse_center:
                delta *= 0.1
            node.position += delta * 0.7

    def _store_time_slice(self):
        slice = {
            "step": self.simulation_step,
            "phases": [n.phase for n in self.nodes],
            "entropy": [n.entropy for n in self.nodes],
            "resonant_nodes": [i for i, n in enumerate(self.nodes) if n.is_resonant],
            "collapse_zones": self.collapse_zones.copy(),
            "glyphs": self.glyphs.copy(),
        }
        self.time_slices.append(slice)
        if len(self.time_slices) > 20:
            self.time_slices.pop(0)


class SEMFVisualizer:
    """Visualization helper for SEMF lattice."""

    def __init__(self, lattice: SEMFLattice):
        self.lattice = lattice
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.scatter = None
        self.lines = []

    def setup_plot(self):
        self.ax.clear()
        max_dim = max(self.lattice.dimensions) * self.lattice.spacing
        self.ax.set_xlim(0, max_dim)
        self.ax.set_ylim(0, max_dim)
        self.ax.set_zlim(0, max_dim)
        self.ax.set_title(f"SEMF Lattice - Step {self.lattice.simulation_step}")
        pos = np.array([n.position for n in self.lattice.nodes])
        self.scatter = self.ax.scatter(
            pos[:, 0], pos[:, 1], pos[:, 2],
            c=self._get_node_colors(), s=self._get_node_sizes(), cmap="viridis", alpha=0.8
        )
        return self.scatter,

    def _get_node_colors(self):
        colors = []
        for i, node in enumerate(self.lattice.nodes):
            base = (node.phase % (2 * np.pi)) / (2 * np.pi)
            if i in self.lattice.glyphs:
                colors.append(1.0)
            elif node.is_collapse_center:
                colors.append(0.0)
            elif node.is_resonant:
                colors.append(0.9)
            else:
                colors.append(base)
        return colors

    def _get_node_sizes(self):
        sizes = []
        for i, node in enumerate(self.lattice.nodes):
            base = 30
            if i in self.lattice.glyphs:
                sizes.append(base * 2.0)
            elif node.is_collapse_center:
                sizes.append(base * 1.5)
            elif node.is_resonant:
                pulse = 1.0 + 0.5 * np.sin(self.lattice.simulation_step * 0.2)
                sizes.append(base * pulse)
            else:
                entropy_factor = 1.0 - min(1.0, node.entropy)
                sizes.append(base * (0.7 + 0.3 * entropy_factor))
        return sizes

    def update(self, frame):
        self.lattice.simulate_step()
        if self.lattice.simulation_step % 5 == 0:
            self.lattice.perform_temporal_resonance_injection()
            self.lattice.calculate_reciprocal_space_contribution()
            self.lattice.detect_glyphs()
        pos = np.array([n.position for n in self.lattice.nodes])
        self.scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        self.scatter.set_array(np.array(self._get_node_colors()))
        self.scatter.set_sizes(self._get_node_sizes())
        self.ax.set_title(f"SEMF Lattice - Step {self.lattice.simulation_step}")
        for line in self.lines:
            line.remove()
        self.lines = []
        if frame % 2 == 0:
            self._draw_entanglement_lines()
        return self.scatter,

    def _draw_entanglement_lines(self):
        max_lines = 100
        all_lines = []
        for i, node in enumerate(self.lattice.nodes):
            for j in node.entangled_nodes:
                if i < j:
                    other = self.lattice.nodes[j]
                    special = (
                        node.is_resonant or self.lattice.nodes[j].is_resonant or
                        node.is_collapse_center or self.lattice.nodes[j].is_collapse_center or
                        i in self.lattice.glyphs or j in self.lattice.glyphs
                    )
                    all_lines.append((i, j, special))
        special_lines = [ln for ln in all_lines if ln[2]]
        normal_lines = [ln for ln in all_lines if not ln[2]]
        lines_to_draw = list(special_lines)
        if len(lines_to_draw) < max_lines:
            max_normal = max_lines - len(special_lines)
            if normal_lines:
                lines_to_draw.extend(random.sample(normal_lines, min(max_normal, len(normal_lines))))
        for i, j, special in lines_to_draw:
            a = self.lattice.nodes[i].position
            b = self.lattice.nodes[j].position
            color = "red" if special else "gray"
            line = self.ax.plot(
                [a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color=color, linewidth=0.5, alpha=0.6
            )[0]
            self.lines.append(line)


def run_simulation(steps=50, interval=200):
    lattice = SEMFLattice()
    vis = SEMFVisualizer(lattice)
    anim = animation.FuncAnimation(
        vis.fig,
        vis.update,
        init_func=vis.setup_plot,
        frames=steps,
        interval=interval,
        blit=False,
    )
    plt.show()


if __name__ == "__main__":
    run_simulation()
