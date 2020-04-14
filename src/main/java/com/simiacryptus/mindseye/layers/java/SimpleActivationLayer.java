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
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;
import java.util.function.IntFunction;

/**
 * The type Simple activation layer.
 *
 * @param <T> the type parameter
 */
@SuppressWarnings("serial")
public abstract class SimpleActivationLayer<T extends SimpleActivationLayer<T>> extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SigmoidActivationLayer.class);

  /**
   * Instantiates a new Simple activation layer.
   */
  public SimpleActivationLayer() {
    super();
    this.frozen = true;
  }

  /**
   * Instantiates a new Simple activation layer.
   *
   * @param id the id
   */
  protected SimpleActivationLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (1 != inObj.length) {
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException();
    }
    final Result inObj0 = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    final TensorList indata0 = inObj0.getData();
    final int itemCnt = indata0.length();
    assert 0 < itemCnt;
    @Nonnull final Tensor inputGradientA[] = new Tensor[itemCnt];
    TensorArray data = fwd(indata0, inputGradientA);
    final boolean inObj0Alive = inObj0.isAlive();
    Result.Accumulator accumulator = new Accumulator(inObj0Alive, itemCnt, inputGradientA, inObj0.getAccumulator());
    inObj0.freeRef();
    return new Result(data, accumulator, inObj0Alive || !isFrozen());
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SimpleActivationLayer<T> addRef() {
    return (SimpleActivationLayer<T>) super.addRef();
  }

  /**
   * Eval.
   *
   * @param x       the x
   * @param results the results
   */
  protected abstract void eval(final double x, double[] results);

  @NotNull
  private TensorArray fwd(TensorList inputList, @RefIgnore Tensor[] inputGradient_out) {
    return new TensorArray(RefIntStream.range(0, inputList.length()).parallel()
        .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) dataIndex -> {
          @Nullable final Tensor input = inputList.get(dataIndex);
          @Nonnull final Tensor output = new Tensor(input.getDimensions());
          int length = input.length();
          @Nonnull final Tensor inputGradient = new Tensor(length);
          @Nonnull final double[] results = new double[2];
          double[] inputData = input.getData();
          for (int i = 0; i < length; i++) {
            eval(inputData[i], results);
            inputGradient.set(i, results[1]);
            output.set(i, results[0]);
          }
          RefUtil.set(inputGradient_out, dataIndex, inputGradient);
          input.freeRef();
          return output;
        }, inputList))
        .toArray(Tensor[]::new));
  }

  private static class Accumulator extends Result.Accumulator {

    private final boolean inObj0Alive;
    private final int itemCnt;
    private final Tensor[] inputGradientA;
    private Result.Accumulator accumulator;

    /**
     * Instantiates a new Accumulator.
     *
     * @param inObj0Alive    the in obj 0 alive
     * @param itemCnt        the item cnt
     * @param inputGradientA the input gradient a
     * @param accumulator    the accumulator
     */
    public Accumulator(boolean inObj0Alive, int itemCnt, Tensor[] inputGradientA, Result.Accumulator accumulator) {
      this.inObj0Alive = inObj0Alive;
      this.itemCnt = itemCnt;
      this.inputGradientA = inputGradientA;
      this.accumulator = accumulator;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
      data.assertAlive();
      if (null != buffer) buffer.assertAlive();
      if (inObj0Alive) {
        this.accumulator.accept(buffer, new TensorArray(RefIntStream.range(0, itemCnt).parallel()
            .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) dataIndex -> {
              @Nonnull final Tensor passback = new Tensor(data.getDimensions());
              Tensor tensor1 = null == inputGradientA ? null : inputGradientA[dataIndex].addRef();
              @Nullable final double[] gradientData = null == tensor1 ? null : tensor1.getData();
              @Nullable
              Tensor tensor = data.get(dataIndex);
              RefIntStream.range(0, passback.length()).forEach(RefUtil.wrapInterface(i -> {
                final double v = null == gradientData ? 0 : gradientData[i];
                if (Double.isFinite(v)) {
                  passback.set(i, tensor.get(i) * v);
                }
              }, passback.addRef(), tensor));
              if (null != tensor1) tensor1.freeRef();
              return passback;
            }, data, RefUtil.addRef(inputGradientA)))
            .toArray(Tensor[]::new)));
      } else {
        data.freeRef();
        if (null != buffer)
          buffer.freeRef();
      }
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      RefUtil.freeRef(inputGradientA);
      accumulator.freeRef();
    }
  }
}
