module     p2_gg_httbar_d253h4l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d253h4l132.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd253h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd253
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd253h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(36) :: acd253
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd253(1)=dotproduct(ninjaA0,ninjaE3)
      acd253(2)=dotproduct(ninjaE3,spvae1e2)
      acd253(3)=dotproduct(ninjaE3,spvae2k2)
      acd253(4)=abb253(24)
      acd253(5)=dotproduct(ninjaE3,spvae2l4)
      acd253(6)=abb253(73)
      acd253(7)=dotproduct(ninjaE3,spval5e2)
      acd253(8)=dotproduct(ninjaE3,spvae2e1)
      acd253(9)=abb253(46)
      acd253(10)=dotproduct(ninjaE3,spvak2e2)
      acd253(11)=abb253(58)
      acd253(12)=dotproduct(ninjaE3,spvae2k1)
      acd253(13)=dotproduct(ninjaE3,spvak2e1)
      acd253(14)=abb253(7)
      acd253(15)=dotproduct(ninjaE3,spval5e1)
      acd253(16)=abb253(29)
      acd253(17)=dotproduct(ninjaE3,spvae2l3)
      acd253(18)=abb253(22)
      acd253(19)=abb253(28)
      acd253(20)=abb253(65)
      acd253(21)=dotproduct(ninjaE3,spvae1l4)
      acd253(22)=abb253(8)
      acd253(23)=dotproduct(ninjaE3,spvak1e2)
      acd253(24)=abb253(36)
      acd253(25)=dotproduct(ninjaE3,spval3e2)
      acd253(26)=abb253(76)
      acd253(27)=dotproduct(ninjaE3,spvae1k2)
      acd253(28)=abb253(39)
      acd253(29)=abb253(51)
      acd253(30)=abb253(64)
      acd253(31)=acd253(17)*acd253(18)
      acd253(32)=acd253(12)*acd253(14)
      acd253(33)=acd253(3)*acd253(19)
      acd253(31)=acd253(33)+acd253(31)+acd253(32)
      acd253(31)=acd253(13)*acd253(31)
      acd253(32)=acd253(17)*acd253(20)
      acd253(33)=acd253(12)*acd253(16)
      acd253(32)=acd253(32)+acd253(33)
      acd253(32)=acd253(15)*acd253(32)
      acd253(33)=acd253(5)*acd253(6)
      acd253(34)=acd253(3)*acd253(4)
      acd253(33)=acd253(33)+acd253(34)
      acd253(34)=2.0_ki*acd253(1)
      acd253(33)=acd253(33)*acd253(34)
      acd253(31)=acd253(33)+acd253(31)+acd253(32)
      acd253(31)=acd253(2)*acd253(31)
      acd253(32)=acd253(25)*acd253(29)
      acd253(33)=acd253(23)*acd253(28)
      acd253(35)=acd253(10)*acd253(30)
      acd253(32)=acd253(35)+acd253(32)+acd253(33)
      acd253(32)=acd253(27)*acd253(32)
      acd253(33)=-acd253(25)*acd253(26)
      acd253(35)=acd253(23)*acd253(24)
      acd253(36)=-acd253(7)*acd253(22)
      acd253(33)=acd253(36)+acd253(33)+acd253(35)
      acd253(33)=acd253(21)*acd253(33)
      acd253(35)=acd253(10)*acd253(11)
      acd253(36)=acd253(7)*acd253(9)
      acd253(35)=acd253(35)+acd253(36)
      acd253(34)=acd253(35)*acd253(34)
      acd253(32)=acd253(34)+acd253(32)+acd253(33)
      acd253(32)=acd253(8)*acd253(32)
      acd253(31)=acd253(31)+acd253(32)
      brack(ninjaidxt0x0mu0)=acd253(31)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d253h4_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd253h4
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k4
      vecA0(1:4) = - a0(0:3) - qshift(1:4)
      vecA1(1:4) = - a1(0:3)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d253h4l132
