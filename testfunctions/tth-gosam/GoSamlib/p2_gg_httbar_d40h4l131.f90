module     p2_gg_httbar_d40h4l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d40h4l131.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt3mu0 = 0
   integer, parameter :: ninjaidxt2mu0 = 1
   integer, parameter :: ninjaidxt1mu0 = 2
   integer, parameter :: ninjaidxt1mu2 = 3
   integer, parameter :: ninjaidxt0mu0 = 4
   integer, parameter :: ninjaidxt0mu2 = 5
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd40h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(69) :: acd40
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd40(1)=dotproduct(k2,ninjaE3)
      acd40(2)=abb40(16)
      acd40(3)=dotproduct(l4,ninjaE3)
      acd40(4)=abb40(32)
      acd40(5)=dotproduct(ninjaE3,spvae1k2)
      acd40(6)=abb40(15)
      acd40(7)=dotproduct(ninjaE3,spvae2e1)
      acd40(8)=abb40(17)
      acd40(9)=dotproduct(ninjaE3,spval4k2)
      acd40(10)=abb40(18)
      acd40(11)=dotproduct(ninjaE3,spvak2l4)
      acd40(12)=abb40(19)
      acd40(13)=dotproduct(ninjaE3,spvak1l4)
      acd40(14)=abb40(20)
      acd40(15)=dotproduct(ninjaE3,spvae2l4)
      acd40(16)=abb40(21)
      acd40(17)=dotproduct(ninjaE3,spvak1e2)
      acd40(18)=abb40(22)
      acd40(19)=dotproduct(ninjaE3,spval4k1)
      acd40(20)=abb40(23)
      acd40(21)=dotproduct(ninjaE3,spvae1l4)
      acd40(22)=abb40(24)
      acd40(23)=dotproduct(ninjaE3,spvak1k2)
      acd40(24)=abb40(26)
      acd40(25)=dotproduct(ninjaE3,spvae1e2)
      acd40(26)=abb40(29)
      acd40(27)=dotproduct(ninjaE3,spvae2k1)
      acd40(28)=abb40(31)
      acd40(29)=dotproduct(ninjaE3,spval4l5)
      acd40(30)=abb40(33)
      acd40(31)=dotproduct(ninjaE3,spvae2l5)
      acd40(32)=abb40(34)
      acd40(33)=dotproduct(ninjaE3,spval5l4)
      acd40(34)=abb40(36)
      acd40(35)=dotproduct(ninjaE3,spval4e2)
      acd40(36)=abb40(40)
      acd40(37)=dotproduct(ninjaE3,spval5k2)
      acd40(38)=abb40(42)
      acd40(39)=dotproduct(ninjaE3,spvak2e2)
      acd40(40)=abb40(43)
      acd40(41)=dotproduct(ninjaE3,spval5e2)
      acd40(42)=abb40(44)
      acd40(43)=dotproduct(ninjaE3,spval4e1)
      acd40(44)=abb40(55)
      acd40(45)=dotproduct(ninjaE3,spvae2k2)
      acd40(46)=abb40(61)
      acd40(47)=acd40(2)*acd40(1)
      acd40(48)=acd40(4)*acd40(3)
      acd40(49)=acd40(6)*acd40(5)
      acd40(50)=acd40(8)*acd40(7)
      acd40(51)=acd40(10)*acd40(9)
      acd40(52)=acd40(12)*acd40(11)
      acd40(53)=acd40(14)*acd40(13)
      acd40(54)=acd40(16)*acd40(15)
      acd40(55)=acd40(18)*acd40(17)
      acd40(56)=acd40(20)*acd40(19)
      acd40(57)=acd40(22)*acd40(21)
      acd40(58)=acd40(24)*acd40(23)
      acd40(59)=acd40(26)*acd40(25)
      acd40(60)=acd40(28)*acd40(27)
      acd40(61)=acd40(30)*acd40(29)
      acd40(62)=acd40(32)*acd40(31)
      acd40(63)=acd40(34)*acd40(33)
      acd40(64)=acd40(36)*acd40(35)
      acd40(65)=acd40(38)*acd40(37)
      acd40(66)=acd40(40)*acd40(39)
      acd40(67)=acd40(42)*acd40(41)
      acd40(68)=acd40(44)*acd40(43)
      acd40(69)=acd40(46)*acd40(45)
      acd40(47)=acd40(69)+acd40(68)+acd40(67)+acd40(66)+acd40(65)+acd40(64)+acd&
      &40(63)+acd40(62)+acd40(61)+acd40(60)+acd40(59)+acd40(58)+acd40(57)+acd40&
      &(56)+acd40(55)+acd40(54)+acd40(53)+acd40(52)+acd40(51)+acd40(50)+acd40(4&
      &9)+acd40(47)+acd40(48)
      brack(ninjaidxt3mu0)=0.0_ki
      brack(ninjaidxt2mu0)=acd40(47)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d40h4_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd40h4
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      vecA(1:4) = - a(0:3)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d40h4l131
