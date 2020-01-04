/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.layers.StochasticComponent;
import com.simiacryptus.mindseye.network.*;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.*;
import java.util.stream.IntStream;
import com.simiacryptus.ref.wrappers.RefIntStream;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware class StochasticSamplingSubnetLayer extends LayerBase
    implements StochasticComponent {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(StochasticSamplingSubnetLayer.class);

  private final int samples;
  @Nullable
  private final Layer subnetwork;
  private long seed = System.nanoTime();

  public StochasticSamplingSubnetLayer(@org.jetbrains.annotations.Nullable final Layer subnetwork, final int samples) {
    super();
    this.samples = samples;
    this.subnetwork = subnetwork;
  }

  protected StochasticSamplingSubnetLayer(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    super(json);
    samples = json.getAsJsonPrimitive("samples").getAsInt();
    seed = json.getAsJsonPrimitive("seed").getAsLong();
    JsonObject subnetwork = json.getAsJsonObject("subnetwork");
    this.subnetwork = subnetwork == null ? null : Layer.fromJson(subnetwork, rs);
  }

  public long[] getSeeds() {
    Random random = new Random(seed);
    return com.simiacryptus.ref.wrappers.RefIntStream.range(0, this.samples).mapToLong(i -> random.nextLong())
        .toArray();
  }

  @SuppressWarnings("unused")
  public static StochasticSamplingSubnetLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new StochasticSamplingSubnetLayer(json, rs);
  }

  public static Result average(final Result[] samples) {
    PipelineNetwork gateNetwork = new PipelineNetwork(samples.length);
    {
      gateNetwork.add(new SumInputsLayer(), com.simiacryptus.ref.wrappers.RefIntStream.range(0, samples.length)
          .mapToObj(i -> gateNetwork.getInput(i)).toArray(i -> new DAGNode[i]));
      gateNetwork.add(new LinearActivationLayer().setScale(1.0 / samples.length).freeze());
      return gateNetwork.eval(samples);
    }
  }

  @Nullable
  public Result eval(@Nonnull final Result... inObj) {
    if (0 == seed) {
      return subnetwork.eval(inObj);
    }
    Result[] counting = com.simiacryptus.ref.wrappers.RefArrays.stream(inObj).map(r -> {
      return new CountingResult(r, samples);
    }).toArray(i -> new Result[i]);
    return average(com.simiacryptus.ref.wrappers.RefArrays.stream(getSeeds()).mapToObj(seed1 -> {
      shuffleSubnet(seed1);
      return subnetwork.eval(counting);
    }).toArray(i -> new Result[i]));
  }

  public void shuffleSubnet(long seed) {
    if (subnetwork instanceof DAGNetwork) {
      ((DAGNetwork) subnetwork).visitNodes(node -> {
        Layer layer = node.getLayer();
        if (layer instanceof StochasticComponent) {
          ((StochasticComponent) layer).shuffle(seed);
        }
      });
    }
    if (subnetwork instanceof StochasticComponent) {
      ((StochasticComponent) subnetwork).shuffle(seed);
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
      DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
    json.addProperty("samples", samples);
    json.addProperty("seed", seed);
    json.add("subnetwork", subnetwork.getJson(resources, dataSerializer));
    return json;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return new com.simiacryptus.ref.wrappers.RefArrayList<>();
  }

  @Nonnull
  @Override
  public Layer setFrozen(final boolean frozen) {
    subnetwork.setFrozen(frozen);
    return super.setFrozen(frozen);
  }

  @Override
  public void shuffle(final long seed) {
    log.info(String.format("Set %s to random seed %s", getName(), seed));
    this.seed = seed;
  }

  @Override
  public void clearNoise() {
    log.info(String.format("Set %s to random null seed", getName()));
    seed = 0;
  }

  public void _free() {
    super._free();
  }

  public @Override @SuppressWarnings("unused") StochasticSamplingSubnetLayer addRef() {
    return (StochasticSamplingSubnetLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") StochasticSamplingSubnetLayer[] addRefs(
      StochasticSamplingSubnetLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(StochasticSamplingSubnetLayer::addRef)
        .toArray((x) -> new StochasticSamplingSubnetLayer[x]);
  }

  public static @SuppressWarnings("unused") StochasticSamplingSubnetLayer[][] addRefs(
      StochasticSamplingSubnetLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(StochasticSamplingSubnetLayer::addRefs)
        .toArray((x) -> new StochasticSamplingSubnetLayer[x][]);
  }

}
