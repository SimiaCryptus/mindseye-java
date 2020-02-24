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
import com.simiacryptus.mindseye.network.CountingResult;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.Random;
import java.util.function.IntFunction;
import java.util.function.LongFunction;

@SuppressWarnings("serial")
public class StochasticSamplingSubnetLayer extends LayerBase implements StochasticComponent {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(StochasticSamplingSubnetLayer.class);

  private final int samples;
  @Nullable
  private final Layer subnetwork;
  private long seed = RefSystem.nanoTime();

  public StochasticSamplingSubnetLayer(@Nullable final Layer subnetwork, final int samples) {
    super();
    this.samples = samples;
    this.subnetwork = subnetwork;
  }

  protected StochasticSamplingSubnetLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    samples = json.getAsJsonPrimitive("samples").getAsInt();
    seed = json.getAsJsonPrimitive("seed").getAsLong();
    JsonObject subnetwork = json.getAsJsonObject("subnetwork");
    this.subnetwork = Layer.fromJson(subnetwork, rs);
  }

  public long[] getSeeds() {
    Random random = new Random(seed);
    return RefIntStream.range(0, this.samples).mapToLong(i -> random.nextLong()).toArray();
  }

  @Nonnull
  @Override
  public void setFrozen(final boolean frozen) {
    assert subnetwork != null;
    subnetwork.setFrozen(frozen);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static StochasticSamplingSubnetLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new StochasticSamplingSubnetLayer(json, rs);
  }

  @Nullable
  public static Result average(@Nonnull final Result[] samples) {
    PipelineNetwork gateNetwork = new PipelineNetwork(samples.length);
    RefUtil.freeRef(gateNetwork.add(new SumInputsLayer(),
        RefIntStream.range(0, samples.length)
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends DAGNode>) gateNetwork::getInput,
                gateNetwork.addRef()))
            .toArray(DAGNode[]::new)));
    LinearActivationLayer scale = new LinearActivationLayer();
    scale.setScale(1.0 / samples.length);
    scale.freeze();
    RefUtil.freeRef(gateNetwork.add(scale));
    Result result = gateNetwork.eval(samples);
    gateNetwork.freeRef();
    return result;
  }

  @Nullable
  public Result eval(@Nonnull final Result... inObj) {
    if (0 == seed) {
      assert subnetwork != null;
      return subnetwork.eval(inObj);
    }
    Result[] counting = RefArrays.stream(inObj).map(r -> {
      return new CountingResult(r, samples);
    }).toArray(Result[]::new);
    return average(
        RefArrays.stream(getSeeds()).mapToObj(RefUtil.wrapInterface((LongFunction<? extends Result>) seed1 -> {
          shuffleSubnet(seed1);
          assert subnetwork != null;
          return subnetwork.eval(RefUtil.addRefs(counting));
        }, counting)).toArray(Result[]::new));
  }

  public void shuffleSubnet(long seed) {
    if (subnetwork instanceof DAGNetwork) {
      ((DAGNetwork) subnetwork).visitNodes(node -> {
        Layer layer = node.getLayer();
        node.freeRef();
        if (layer instanceof StochasticComponent) {
          ((StochasticComponent) layer).shuffle(seed);
        }
        if (null != layer)
          layer.freeRef();
      });
    }
    if (subnetwork instanceof StochasticComponent) {
      ((StochasticComponent) subnetwork).shuffle(seed);
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("samples", samples);
    json.addProperty("seed", seed);
    assert subnetwork != null;
    json.add("subnetwork", subnetwork.getJson(resources, dataSerializer));
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return new RefArrayList<>();
  }

  @Override
  public void shuffle(final long seed) {
    log.info(RefString.format("Set %s to random seed %s", getName(), seed));
    this.seed = seed;
  }

  @Override
  public void clearNoise() {
    log.info(RefString.format("Set %s to random null seed", getName()));
    seed = 0;
  }

  public void _free() {
    if (null != subnetwork)
      subnetwork.freeRef();
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  StochasticSamplingSubnetLayer addRef() {
    return (StochasticSamplingSubnetLayer) super.addRef();
  }

}
