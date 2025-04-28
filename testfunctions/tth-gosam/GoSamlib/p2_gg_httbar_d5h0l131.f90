module     p2_gg_httbar_d5h0l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d5h0l131.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt1mu0 = 0
   integer, parameter :: ninjaidxt0mu0 = 1
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd5h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd5
      complex(ki), dimension (0:*), intent(inout) :: brack
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd5h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(39) :: acd5
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd5(1)=dotproduct(ninjaA,ninjaE3)
      acd5(2)=abb5(15)
      acd5(3)=dotproduct(ninjaE3,spval4l3)
      acd5(4)=abb5(6)
      acd5(5)=dotproduct(ninjaE3,spval5l3)
      acd5(6)=abb5(7)
      acd5(7)=dotproduct(ninjaE3,spvae1e2)
      acd5(8)=abb5(9)
      acd5(9)=dotproduct(ninjaE3,spvak2e1)
      acd5(10)=abb5(10)
      acd5(11)=dotproduct(ninjaE3,spvae1l3)
      acd5(12)=abb5(11)
      acd5(13)=dotproduct(ninjaE3,spval4k2)
      acd5(14)=abb5(12)
      acd5(15)=dotproduct(ninjaE3,spvae2e1)
      acd5(16)=abb5(13)
      acd5(17)=dotproduct(ninjaE3,spvak2e2)
      acd5(18)=abb5(14)
      acd5(19)=dotproduct(ninjaE3,spvae2l3)
      acd5(20)=abb5(16)
      acd5(21)=dotproduct(ninjaE3,spval3e2)
      acd5(22)=abb5(17)
      acd5(23)=dotproduct(ninjaE3,spval3e1)
      acd5(24)=abb5(18)
      acd5(25)=dotproduct(ninjaE3,spval5k2)
      acd5(26)=abb5(20)
      acd5(27)=acd5(2)*acd5(1)
      acd5(28)=acd5(4)*acd5(3)
      acd5(29)=acd5(6)*acd5(5)
      acd5(30)=acd5(8)*acd5(7)
      acd5(31)=acd5(10)*acd5(9)
      acd5(32)=acd5(12)*acd5(11)
      acd5(33)=acd5(14)*acd5(13)
      acd5(34)=acd5(16)*acd5(15)
      acd5(35)=acd5(18)*acd5(17)
      acd5(36)=acd5(20)*acd5(19)
      acd5(37)=acd5(22)*acd5(21)
      acd5(38)=acd5(24)*acd5(23)
      acd5(39)=acd5(26)*acd5(25)
      acd5(27)=acd5(39)+acd5(38)+acd5(37)+acd5(36)+acd5(35)+acd5(34)+acd5(33)+a&
      &cd5(32)+acd5(31)+acd5(30)+acd5(29)+2.0_ki*acd5(27)+acd5(28)
      brack(ninjaidxt1mu0)=0.0_ki
      brack(ninjaidxt0mu0)=acd5(27)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d5h0_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd5h0
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k3+k5
      vecA(1:4) = + a(0:3) - qshift(1:4)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-2))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d5h0l131
