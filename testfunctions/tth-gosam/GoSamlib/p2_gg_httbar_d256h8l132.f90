module     p2_gg_httbar_d256h8l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d256h8l132.f90
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
      use p2_gg_httbar_abbrevd256h8
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd256
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
      use p2_gg_httbar_abbrevd256h8
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(72) :: acd256
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd256(1)=dotproduct(k2,ninjaE3)
      acd256(2)=dotproduct(e1,ninjaE3)
      acd256(3)=dotproduct(e2,ninjaE3)
      acd256(4)=abb256(93)
      acd256(5)=dotproduct(ninjaE3,spvae2k2)
      acd256(6)=abb256(159)
      acd256(7)=dotproduct(ninjaE3,spvak2e2)
      acd256(8)=abb256(147)
      acd256(9)=dotproduct(ninjaE3,spvae1k2)
      acd256(10)=abb256(63)
      acd256(11)=dotproduct(ninjaE3,spvak2e1)
      acd256(12)=abb256(101)
      acd256(13)=dotproduct(l4,ninjaE3)
      acd256(14)=abb256(140)
      acd256(15)=dotproduct(ninjaE3,spval4e2)
      acd256(16)=abb256(21)
      acd256(17)=dotproduct(ninjaE3,spval4e1)
      acd256(18)=abb256(157)
      acd256(19)=dotproduct(l5,ninjaE3)
      acd256(20)=abb256(137)
      acd256(21)=dotproduct(ninjaE3,spvae2l5)
      acd256(22)=abb256(114)
      acd256(23)=dotproduct(ninjaE3,spvae1l5)
      acd256(24)=abb256(66)
      acd256(25)=dotproduct(ninjaA0,ninjaE3)
      acd256(26)=abb256(65)
      acd256(27)=dotproduct(ninjaE3,spvak2l5)
      acd256(28)=abb256(79)
      acd256(29)=dotproduct(ninjaE3,spval4l5)
      acd256(30)=abb256(38)
      acd256(31)=dotproduct(ninjaE3,spval4k2)
      acd256(32)=abb256(41)
      acd256(33)=dotproduct(ninjaE3,spvak1k2)
      acd256(34)=abb256(45)
      acd256(35)=dotproduct(ninjaE3,spvak2k1)
      acd256(36)=abb256(81)
      acd256(37)=dotproduct(ninjaE3,spvak1l5)
      acd256(38)=abb256(106)
      acd256(39)=dotproduct(ninjaE3,spval4k1)
      acd256(40)=abb256(110)
      acd256(41)=abb256(124)
      acd256(42)=abb256(131)
      acd256(43)=abb256(52)
      acd256(44)=abb256(112)
      acd256(45)=abb256(7)
      acd256(46)=dotproduct(ninjaE3,spval5e2)
      acd256(47)=abb256(130)
      acd256(48)=dotproduct(ninjaE3,spvae2l4)
      acd256(49)=dotproduct(ninjaE3,spvae2k1)
      acd256(50)=dotproduct(ninjaE3,spvak1e2)
      acd256(51)=abb256(99)
      acd256(52)=abb256(133)
      acd256(53)=abb256(163)
      acd256(54)=abb256(107)
      acd256(55)=abb256(13)
      acd256(56)=dotproduct(ninjaE3,spval5e1)
      acd256(57)=dotproduct(ninjaE3,spvae1l4)
      acd256(58)=abb256(165)
      acd256(59)=dotproduct(ninjaE3,spvae1k1)
      acd256(60)=dotproduct(ninjaE3,spvak1e1)
      acd256(61)=acd256(37)*acd256(38)
      acd256(62)=acd256(35)*acd256(36)
      acd256(63)=acd256(29)*acd256(30)
      acd256(64)=acd256(39)*acd256(40)
      acd256(65)=acd256(33)*acd256(34)
      acd256(66)=acd256(19)*acd256(20)
      acd256(67)=acd256(13)*acd256(14)
      acd256(68)=acd256(31)*acd256(32)
      acd256(69)=acd256(27)*acd256(28)
      acd256(70)=acd256(1)*acd256(4)
      acd256(71)=2.0_ki*acd256(25)
      acd256(72)=acd256(26)*acd256(71)
      acd256(61)=acd256(72)+acd256(70)+acd256(69)+acd256(68)+acd256(67)+acd256(&
      &66)+acd256(65)+acd256(64)+acd256(63)+acd256(61)+acd256(62)
      acd256(61)=acd256(3)*acd256(61)
      acd256(62)=acd256(21)*acd256(44)
      acd256(63)=-acd256(15)*acd256(42)
      acd256(64)=acd256(7)*acd256(43)
      acd256(65)=-acd256(5)*acd256(41)
      acd256(62)=acd256(65)+acd256(64)+acd256(62)+acd256(63)
      acd256(62)=acd256(62)*acd256(71)
      acd256(63)=-acd256(41)*acd256(48)
      acd256(64)=-acd256(7)*acd256(47)
      acd256(63)=acd256(63)+acd256(64)
      acd256(63)=acd256(31)*acd256(63)
      acd256(64)=-acd256(43)*acd256(46)
      acd256(65)=acd256(5)*acd256(45)
      acd256(64)=acd256(64)+acd256(65)
      acd256(64)=acd256(27)*acd256(64)
      acd256(65)=acd256(7)*acd256(8)
      acd256(66)=acd256(5)*acd256(6)
      acd256(65)=acd256(65)+acd256(66)
      acd256(65)=acd256(1)*acd256(65)
      acd256(66)=acd256(39)*acd256(42)*acd256(50)
      acd256(67)=acd256(33)*acd256(41)*acd256(49)
      acd256(68)=acd256(19)*acd256(21)*acd256(22)
      acd256(69)=-acd256(13)*acd256(15)*acd256(16)
      acd256(61)=acd256(61)+acd256(62)+acd256(65)+acd256(64)+acd256(63)+acd256(&
      &69)+acd256(68)+acd256(66)+acd256(67)
      acd256(61)=acd256(2)*acd256(61)
      acd256(62)=acd256(23)*acd256(52)
      acd256(63)=-acd256(17)*acd256(54)
      acd256(64)=acd256(11)*acd256(53)
      acd256(65)=-acd256(9)*acd256(51)
      acd256(62)=acd256(65)+acd256(64)+acd256(62)+acd256(63)
      acd256(62)=acd256(62)*acd256(71)
      acd256(63)=-acd256(51)*acd256(57)
      acd256(64)=-acd256(11)*acd256(58)
      acd256(63)=acd256(63)+acd256(64)
      acd256(63)=acd256(31)*acd256(63)
      acd256(64)=-acd256(53)*acd256(56)
      acd256(65)=acd256(9)*acd256(55)
      acd256(64)=acd256(64)+acd256(65)
      acd256(64)=acd256(27)*acd256(64)
      acd256(65)=-acd256(11)*acd256(12)
      acd256(66)=acd256(9)*acd256(10)
      acd256(65)=acd256(65)+acd256(66)
      acd256(65)=acd256(1)*acd256(65)
      acd256(66)=acd256(39)*acd256(54)*acd256(60)
      acd256(67)=acd256(33)*acd256(51)*acd256(59)
      acd256(68)=-acd256(19)*acd256(23)*acd256(24)
      acd256(69)=acd256(13)*acd256(17)*acd256(18)
      acd256(62)=acd256(62)+acd256(65)+acd256(64)+acd256(63)+acd256(69)+acd256(&
      &68)+acd256(66)+acd256(67)
      acd256(62)=acd256(3)*acd256(62)
      acd256(61)=acd256(62)+acd256(61)
      brack(ninjaidxt0x0mu0)=acd256(61)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d256h8_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd256h8
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2
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
end module     p2_gg_httbar_d256h8l132
