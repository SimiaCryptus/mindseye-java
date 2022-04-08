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
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

/**
 * A class that represents a static scalar loss layer.
 *
 * @author John Doe
 * @docgenVersion 9
 * @since 1.0
 */
@SuppressWarnings("serial")
public class StaticScalarLossLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(StaticScalarLossLayer.class);
  private double target = 0.0;

  /**
   * Instantiates a new Static scalar loss layer.
   */
  public StaticScalarLossLayer() {
  }

  /**
   * Instantiates a new Static scalar loss layer.
   *
   * @param id the id
   */
  protected StaticScalarLossLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  /**
   * Returns the target value.
   *
   * @return the target value
   * @docgenVersion 9
   */
  public double getTarget() {
    return target;
  }

  /**
   * Sets the target value.
   *
   * @param target the target value
   * @docgenVersion 9
   */
  public void setTarget(double target) {
    this.target = target;
  }

  /**
   * Creates a new {@link StaticScalarLossLayer} from a JSON object.
   *
   * @param json the JSON object to use
   * @param rs   the map of character sequences to byte arrays
   * @return the new {@link StaticScalarLossLayer}
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static StaticScalarLossLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new StaticScalarLossLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (1 != inObj.length) {
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException();
    }
    //if (inObj[0].getData().length() != 1) throw new IllegalArgumentException();
    final Result in0 = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    boolean alive = in0.isAlive();
    TensorArray data = fwd(in0.getData());
    Result.Accumulator accumulator = new Accumulator(this.getTarget(), in0.getAccumulator(), in0.isAlive(), in0.getData());
    in0.freeRef();
    return new Result(data, accumulator, alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  /**
   * This method frees the object.
   *
   * @docgenVersion 9
   */
  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  StaticScalarLossLayer addRef() {
    return (StaticScalarLossLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(TensorList indata) {
    return new TensorArray(RefIntStream.range(0, indata.length()).parallel()
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          @Nullable final Tensor a = indata.get(dataIndex);
          final double diff = Math.abs(a.get(0) - getTarget());
          a.freeRef();
          return new Tensor(new double[]{diff}, 1);
        }, indata)).toArray(Tensor[]::new));
  }

  /**
   * The Accumulator class is used to accumulate data from a TensorList.
   *
   * @param indata      The data to be accumulated.
   * @param target      The target value to accumulate to.
   * @param accumulator The accumulator to use.
   * @param alive       Whether or not the accumulator is alive.
   * @docgenVersion 9
   */
  private static class Accumulator extends Result.Accumulator {

    private final TensorList indata;
    private double target;
    private Result.Accumulator accumulator;
    private boolean alive;

    /**
     * Instantiates a new Accumulator.
     *
     * @param target      the target
     * @param accumulator the accumulator
     * @param alive       the alive
     * @param indata      the indata
     */
    public Accumulator(double target, Result.Accumulator accumulator, boolean alive, @NotNull TensorList indata) {
      this.indata = indata;
      this.target = target;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
      if (alive) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(RefIntStream.range(0, data.length()).parallel()
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
              @Nullable final Tensor a = indata.get(dataIndex);
              Tensor tensor = data.get(dataIndex);
              final double deriv = tensor.get(0)
                  * (a.get(0) - target < 0 ? -1 : 1);
              tensor.freeRef();
              a.freeRef();
              return new Tensor(new double[]{deriv}, 1);
            }, indata.addRef(), data.addRef()))
            .toArray(Tensor[]::new));
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        this.accumulator.accept(buffer1, tensorArray);
      }
      data.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    /**
     * Frees resources used by this object.
     *
     * @docgenVersion 9
     */
    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      indata.freeRef();
      accumulator.freeRef();
    }
  }
}
