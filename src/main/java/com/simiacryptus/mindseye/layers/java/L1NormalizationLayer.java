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
import com.simiacryptus.ref.wrappers.RefDoubleStream;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.util.ArrayUtil;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

/**
 * The L1NormalizationLayer class is a Java class that contains a logger and a double variable.
 * The double variable, maxInput, is set to 50.
 *
 * @docgenVersion 9
 */
@SuppressWarnings("serial")
public class L1NormalizationLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(L1NormalizationLayer.class);
  /**
   * The Max input.
   */
  double maxInput = 50;

  /**
   * Instantiates a new L 1 normalization layer.
   */
  public L1NormalizationLayer() {
  }

  /**
   * Instantiates a new L 1 normalization layer.
   *
   * @param id the id
   */
  protected L1NormalizationLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  /**
   * Creates a new {@link L1NormalizationLayer} from a JSON object.
   *
   * @param json the JSON object to use for creating the layer
   * @param rs   a map of character sequences to byte arrays
   * @return a new {@link L1NormalizationLayer}
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static L1NormalizationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new L1NormalizationLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... input) {
    final Result in = input[0].addRef();
    RefUtil.freeRef(input);
    final TensorList inData = in.getData();
    boolean alive = in.isAlive();
    Result.Accumulator accumulator = new Accumulator(inData.addRef(), in.getAccumulator(), alive);
    in.freeRef();
    TensorArray data = fwd(inData);
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
  L1NormalizationLayer addRef() {
    return (L1NormalizationLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(TensorList inData) {
    return new TensorArray(RefIntStream.range(0, inData.length())
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          @Nullable final Tensor value = inData.get(dataIndex);
          final double sum = value.sum();
          if (!Double.isFinite(sum) || 0 == sum) {
            return value;
          } else {
            Tensor temp_26_0003 = value.scale(1.0 / sum);
            value.freeRef();
            return temp_26_0003;
          }
        }, inData)).toArray(Tensor[]::new));
  }

  /**
   * The Accumulator class is used to accumulate the results of a TensorList.
   *
   * @author Author Name
   * @version 1.0
   * @docgenVersion 9
   * @since 1.0
   */
  private static class Accumulator extends Result.Accumulator {

    private final TensorList inData;
    private Result.Accumulator accumulator;
    private boolean alive;

    /**
     * Instantiates a new Accumulator.
     *
     * @param inData      the in data
     * @param accumulator the accumulator
     * @param alive       the alive
     */
    public Accumulator(TensorList inData, Result.Accumulator accumulator, boolean alive) {
      this.inData = inData;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList outDelta) {
      if (alive) {
        final Tensor[] passbackArray = RefIntStream.range(0, outDelta.length())
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
              Tensor inputTensor = inData.get(dataIndex);
              @Nullable final double[] value = inputTensor.getData();
              inputTensor.freeRef();
              Tensor outputTensor = outDelta.get(dataIndex);
              @Nullable final double[] delta = outputTensor.getData();
              final double dot = ArrayUtil.dot(value, delta);
              final double sum = RefArrays.stream(value).sum();
              @Nonnull final Tensor passback = new Tensor(outputTensor.getDimensions());
              outputTensor.freeRef();
              @Nullable final double[] passbackData = passback.getData();
              if (0 != sum || Double.isFinite(sum)) {
                for (int i = 0; i < value.length; i++) {
                  passbackData[i] = (delta[i] - dot / sum) / sum;
                }
              }
              return passback;
            }, inData.addRef(), outDelta.addRef()))
            .toArray(Tensor[]::new);
        assert RefArrays.stream(RefUtil.addRef(passbackArray)).flatMapToDouble(x -> {
          RefDoubleStream temp_26_0004 = RefArrays.stream(x.getData());
          x.freeRef();
          return temp_26_0004;
        }).allMatch(Double::isFinite);
        @Nonnull
        TensorArray tensorArray = new TensorArray(RefUtil.addRef(passbackArray));
        RefUtil.freeRef(passbackArray);
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        this.accumulator.accept(buffer1, tensorArray);
      }
      outDelta.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    /**
     * Frees resources.
     *
     * @docgenVersion 9
     */
    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
      inData.freeRef();
    }
  }
}
