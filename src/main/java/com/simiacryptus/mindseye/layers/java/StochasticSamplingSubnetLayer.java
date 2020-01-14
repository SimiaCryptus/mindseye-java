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
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
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
    Layer temp_22_0001 = subnetwork == null ? null : subnetwork.addRef();
    this.subnetwork = temp_22_0001 == null ? null : temp_22_0001.addRef();
    if (null != temp_22_0001)
      temp_22_0001.freeRef();
    if (null != subnetwork)
      subnetwork.freeRef();
  }

  protected StochasticSamplingSubnetLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    samples = json.getAsJsonPrimitive("samples").getAsInt();
    seed = json.getAsJsonPrimitive("seed").getAsLong();
    JsonObject subnetwork = json.getAsJsonObject("subnetwork");
    Layer temp_22_0008 = Layer.fromJson(subnetwork, rs);
    Layer temp_22_0002 = temp_22_0008.addRef();
    temp_22_0008.freeRef();
    this.subnetwork = temp_22_0002 == null ? null : temp_22_0002.addRef();
    if (null != temp_22_0002)
      temp_22_0002.freeRef();
  }

  public long[] getSeeds() {
    Random random = new Random(seed);
    return RefIntStream.range(0, this.samples).mapToLong(i -> random.nextLong()).toArray();
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
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends DAGNode>) i -> gateNetwork.getInput(i),
                gateNetwork.addRef()))
            .toArray(i -> new DAGNode[i])));
    LinearActivationLayer temp_22_0007 = new LinearActivationLayer();
    LinearActivationLayer temp_22_0009 = temp_22_0007.setScale(1.0 / samples.length);
    RefUtil.freeRef(gateNetwork.add(temp_22_0009.freeze()));
    temp_22_0009.freeRef();
    temp_22_0007.freeRef();
    Result temp_22_0003 = gateNetwork.eval(Result.addRefs(samples));
    gateNetwork.freeRef();
    ReferenceCounting.freeRefs(samples);
    return temp_22_0003;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  StochasticSamplingSubnetLayer[] addRefs(
      @Nullable StochasticSamplingSubnetLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(StochasticSamplingSubnetLayer::addRef)
        .toArray((x) -> new StochasticSamplingSubnetLayer[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  StochasticSamplingSubnetLayer[][] addRefs(
      @Nullable StochasticSamplingSubnetLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(StochasticSamplingSubnetLayer::addRefs)
        .toArray((x) -> new StochasticSamplingSubnetLayer[x][]);
  }

  @Nullable
  public Result eval(@Nonnull final Result... inObj) {
    if (0 == seed) {
      assert subnetwork != null;
      Result temp_22_0006 = subnetwork.eval(Result.addRefs(inObj));
      ReferenceCounting.freeRefs(inObj);
      return temp_22_0006;
    }
    Result[] counting = RefArrays.stream(Result.addRefs(inObj)).map(r -> {
      CountingResult temp_22_0004 = new CountingResult(r == null ? null : r.addRef(), samples);
      if (null != r)
        r.freeRef();
      return temp_22_0004;
    }).toArray(i -> new Result[i]);
    ReferenceCounting.freeRefs(inObj);
    Result temp_22_0005 = average(
        RefArrays.stream(getSeeds()).mapToObj(RefUtil.wrapInterface((LongFunction<? extends Result>) seed1 -> {
          shuffleSubnet(seed1);
          assert subnetwork != null;
          return subnetwork.eval(Result.addRefs(counting));
        }, Result.addRefs(counting))).toArray(i -> new Result[i]));
    ReferenceCounting.freeRefs(counting);
    return temp_22_0005;
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

  @Nonnull
  @Override
  public Layer setFrozen(final boolean frozen) {
    assert subnetwork != null;
    RefUtil.freeRef(subnetwork.setFrozen(frozen));
    return super.setFrozen(frozen);
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
