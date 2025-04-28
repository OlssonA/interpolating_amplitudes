module     p2_gg_httbar_d255h8l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d255h8l132_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt1x0mu0 = 0
   integer, parameter :: ninjaidxt0x0mu0 = 1
   integer, parameter :: ninjaidxt0x1mu0 = 2
   public :: numerator_t2
contains
!---#[ subroutine brack_21:
   pure subroutine brack_21(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd255h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd255
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      brack(ninjaidxt1x0mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd255h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(29) :: acd255
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd255(1)=dotproduct(ninjaA0,ninjaE3)
      acd255(2)=dotproduct(ninjaE3,spvak2e1)
      acd255(3)=dotproduct(ninjaE3,spvae1e2)
      acd255(4)=abb255(7)
      acd255(5)=dotproduct(ninjaE3,spval4e1)
      acd255(6)=abb255(78)
      acd255(7)=dotproduct(ninjaE3,spvae1k2)
      acd255(8)=dotproduct(ninjaE3,spvae2e1)
      acd255(9)=abb255(54)
      acd255(10)=dotproduct(ninjaE3,spvae1l5)
      acd255(11)=abb255(30)
      acd255(12)=dotproduct(ninjaE3,spvae2k2)
      acd255(13)=abb255(14)
      acd255(14)=dotproduct(ninjaE3,spval3e1)
      acd255(15)=abb255(40)
      acd255(16)=dotproduct(ninjaE3,spvae2l5)
      acd255(17)=abb255(82)
      acd255(18)=dotproduct(ninjaE3,spvak2e2)
      acd255(19)=abb255(19)
      acd255(20)=dotproduct(ninjaE3,spvae1l3)
      acd255(21)=abb255(53)
      acd255(22)=dotproduct(ninjaE3,spval4e2)
      acd255(23)=abb255(76)
      acd255(24)=abb255(56)
      acd255(25)=acd255(20)*acd255(24)
      acd255(26)=acd255(10)*acd255(23)
      acd255(25)=acd255(26)+acd255(25)
      acd255(25)=acd255(22)*acd255(25)
      acd255(26)=acd255(20)*acd255(21)
      acd255(27)=acd255(7)*acd255(19)
      acd255(26)=acd255(27)+acd255(26)
      acd255(26)=acd255(18)*acd255(26)
      acd255(27)=-acd255(10)*acd255(11)
      acd255(28)=acd255(7)*acd255(9)
      acd255(27)=acd255(27)+acd255(28)
      acd255(28)=2.0_ki*acd255(1)
      acd255(27)=acd255(27)*acd255(28)
      acd255(25)=acd255(27)+acd255(26)+acd255(25)
      acd255(25)=acd255(8)*acd255(25)
      acd255(26)=acd255(14)*acd255(15)
      acd255(27)=acd255(2)*acd255(13)
      acd255(26)=acd255(27)+acd255(26)
      acd255(26)=acd255(12)*acd255(26)
      acd255(27)=acd255(5)*acd255(6)
      acd255(29)=acd255(2)*acd255(4)
      acd255(27)=acd255(27)+acd255(29)
      acd255(27)=acd255(27)*acd255(28)
      acd255(28)=acd255(14)*acd255(16)*acd255(17)
      acd255(26)=acd255(27)+acd255(28)+acd255(26)
      acd255(26)=acd255(3)*acd255(26)
      acd255(25)=acd255(26)+acd255(25)
      brack(ninjaidxt0x0mu0)=acd255(25)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d255h8_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd255h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k3-k5
      vecA0(1:4) = + a0(0:3) - qshift(1:4)
      vecA1(1:4) = + a1(0:3)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d255h8l132_qp
