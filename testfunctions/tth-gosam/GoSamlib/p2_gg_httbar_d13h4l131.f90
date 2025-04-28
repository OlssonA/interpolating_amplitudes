module     p2_gg_httbar_d13h4l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d13h4l131.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt2mu0 = 0
   integer, parameter :: ninjaidxt1mu0 = 1
   integer, parameter :: ninjaidxt0mu0 = 2
   integer, parameter :: ninjaidxt0mu2 = 3
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd13h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(12) :: acd13
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd13(1)=dotproduct(k1,ninjaE3)
      acd13(2)=dotproduct(k2,ninjaE3)
      acd13(3)=abb13(34)
      acd13(4)=dotproduct(ninjaE3,spval5l4)
      acd13(5)=abb13(16)
      acd13(6)=dotproduct(ninjaE3,spval3l4)
      acd13(7)=abb13(19)
      acd13(8)=dotproduct(ninjaE3,spvak2l3)
      acd13(9)=abb13(23)
      acd13(10)=acd13(8)*acd13(9)
      acd13(11)=acd13(6)*acd13(7)
      acd13(12)=acd13(4)*acd13(5)
      acd13(10)=acd13(12)+acd13(10)-acd13(11)
      acd13(11)=-acd13(1)*acd13(10)
      acd13(12)=acd13(2)-acd13(1)
      acd13(12)=acd13(3)*acd13(12)
      acd13(10)=acd13(12)+acd13(10)
      acd13(10)=acd13(2)*acd13(10)
      acd13(10)=acd13(11)+acd13(10)
      brack(ninjaidxt2mu0)=acd13(10)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd13h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(89) :: acd13
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd13(1)=dotproduct(k1,ninjaE3)
      acd13(2)=dotproduct(k2,ninjaE4)
      acd13(3)=abb13(34)
      acd13(4)=dotproduct(ninjaE4,spval5l4)
      acd13(5)=abb13(16)
      acd13(6)=dotproduct(ninjaE4,spval3l4)
      acd13(7)=abb13(19)
      acd13(8)=dotproduct(ninjaE4,spvak2l3)
      acd13(9)=abb13(23)
      acd13(10)=dotproduct(k1,ninjaE4)
      acd13(11)=dotproduct(k2,ninjaE3)
      acd13(12)=dotproduct(ninjaE3,spval5l4)
      acd13(13)=dotproduct(ninjaE3,spval3l4)
      acd13(14)=dotproduct(ninjaE3,spvak2l3)
      acd13(15)=abb13(15)
      acd13(16)=dotproduct(k1,ninjaA)
      acd13(17)=dotproduct(k2,ninjaA)
      acd13(18)=dotproduct(ninjaA,spval5l4)
      acd13(19)=dotproduct(ninjaA,spval3l4)
      acd13(20)=dotproduct(ninjaA,spvak2l3)
      acd13(21)=abb13(11)
      acd13(22)=abb13(9)
      acd13(23)=dotproduct(l4,ninjaE3)
      acd13(24)=abb13(17)
      acd13(25)=dotproduct(ninjaA,ninjaE3)
      acd13(26)=dotproduct(ninjaE3,spvak2l4)
      acd13(27)=abb13(10)
      acd13(28)=dotproduct(ninjaE3,spvak1k2)
      acd13(29)=abb13(12)
      acd13(30)=dotproduct(ninjaE3,spvak1l4)
      acd13(31)=abb13(13)
      acd13(32)=abb13(20)
      acd13(33)=dotproduct(ninjaE3,spvak1l3)
      acd13(34)=abb13(18)
      acd13(35)=abb13(39)
      acd13(36)=dotproduct(ninjaE3,spval3k1)
      acd13(37)=abb13(21)
      acd13(38)=dotproduct(ninjaE3,spval3k2)
      acd13(39)=abb13(22)
      acd13(40)=abb13(25)
      acd13(41)=dotproduct(ninjaE3,spvak2k1)
      acd13(42)=abb13(26)
      acd13(43)=dotproduct(ninjaE3,spval5k1)
      acd13(44)=abb13(30)
      acd13(45)=dotproduct(ninjaE3,spval5k2)
      acd13(46)=abb13(32)
      acd13(47)=dotproduct(ninjaE3,spval4l3)
      acd13(48)=abb13(36)
      acd13(49)=dotproduct(ninjaE3,spval4k2)
      acd13(50)=abb13(38)
      acd13(51)=dotproduct(l4,ninjaA)
      acd13(52)=dotproduct(ninjaA,ninjaA)
      acd13(53)=dotproduct(ninjaA,spvak2l4)
      acd13(54)=dotproduct(ninjaA,spvak1k2)
      acd13(55)=dotproduct(ninjaA,spvak1l4)
      acd13(56)=dotproduct(ninjaA,spvak1l3)
      acd13(57)=dotproduct(ninjaA,spval3k1)
      acd13(58)=dotproduct(ninjaA,spval3k2)
      acd13(59)=dotproduct(ninjaA,spvak2k1)
      acd13(60)=dotproduct(ninjaA,spval5k1)
      acd13(61)=dotproduct(ninjaA,spval5k2)
      acd13(62)=dotproduct(ninjaA,spval4l3)
      acd13(63)=dotproduct(ninjaA,spval4k2)
      acd13(64)=abb13(14)
      acd13(65)=-acd13(10)+2.0_ki*acd13(2)
      acd13(65)=acd13(65)*acd13(3)
      acd13(66)=acd13(9)*acd13(8)
      acd13(67)=acd13(7)*acd13(6)
      acd13(68)=acd13(5)*acd13(4)
      acd13(65)=acd13(68)+acd13(65)+acd13(66)-acd13(67)
      acd13(65)=acd13(65)*acd13(11)
      acd13(66)=acd13(2)-acd13(10)
      acd13(67)=-acd13(12)*acd13(66)
      acd13(68)=acd13(1)*acd13(4)
      acd13(67)=acd13(67)+acd13(68)
      acd13(67)=acd13(67)*acd13(5)
      acd13(68)=-acd13(13)*acd13(66)
      acd13(69)=acd13(1)*acd13(6)
      acd13(68)=acd13(68)+acd13(69)
      acd13(68)=acd13(68)*acd13(7)
      acd13(66)=-acd13(14)*acd13(66)
      acd13(69)=acd13(1)*acd13(8)
      acd13(66)=acd13(66)+acd13(69)
      acd13(66)=acd13(66)*acd13(9)
      acd13(69)=acd13(2)*acd13(1)*acd13(3)
      acd13(65)=-acd13(65)+acd13(67)-acd13(68)+acd13(66)+acd13(69)-acd13(15)
      acd13(66)=acd13(9)*acd13(20)
      acd13(67)=acd13(7)*acd13(19)
      acd13(68)=acd13(5)*acd13(18)
      acd13(66)=acd13(68)+acd13(66)-acd13(67)
      acd13(67)=-acd13(16)+2.0_ki*acd13(17)
      acd13(67)=acd13(3)*acd13(67)
      acd13(67)=acd13(22)+acd13(67)+acd13(66)
      acd13(67)=acd13(11)*acd13(67)
      acd13(68)=acd13(17)-acd13(16)
      acd13(69)=acd13(14)*acd13(68)
      acd13(70)=-acd13(1)*acd13(20)
      acd13(69)=acd13(70)+acd13(69)
      acd13(69)=acd13(9)*acd13(69)
      acd13(70)=-acd13(13)*acd13(68)
      acd13(71)=acd13(1)*acd13(19)
      acd13(70)=acd13(71)+acd13(70)
      acd13(70)=acd13(7)*acd13(70)
      acd13(71)=acd13(12)*acd13(68)
      acd13(72)=-acd13(1)*acd13(18)
      acd13(71)=acd13(72)+acd13(71)
      acd13(71)=acd13(5)*acd13(71)
      acd13(72)=acd13(50)*acd13(49)
      acd13(73)=acd13(48)*acd13(47)
      acd13(74)=acd13(46)*acd13(45)
      acd13(75)=acd13(44)*acd13(43)
      acd13(76)=acd13(42)*acd13(41)
      acd13(77)=acd13(39)*acd13(38)
      acd13(78)=acd13(37)*acd13(36)
      acd13(79)=acd13(34)*acd13(33)
      acd13(80)=acd13(31)*acd13(30)
      acd13(81)=acd13(29)*acd13(28)
      acd13(82)=acd13(27)*acd13(26)
      acd13(83)=acd13(24)*acd13(23)
      acd13(84)=acd13(15)*acd13(25)
      acd13(85)=acd13(14)*acd13(40)
      acd13(86)=acd13(13)*acd13(35)
      acd13(87)=acd13(12)*acd13(32)
      acd13(88)=acd13(3)*acd13(17)
      acd13(89)=acd13(21)-acd13(88)
      acd13(89)=acd13(1)*acd13(89)
      acd13(67)=acd13(67)+acd13(71)+acd13(70)+acd13(69)+acd13(89)+acd13(87)+acd&
      &13(86)+acd13(85)+2.0_ki*acd13(84)+acd13(83)+acd13(82)+acd13(81)+acd13(80&
      &)+acd13(79)+acd13(78)+acd13(77)+acd13(76)+acd13(75)+acd13(74)+acd13(72)+&
      &acd13(73)
      acd13(69)=-ninjaP*acd13(65)
      acd13(66)=acd13(88)+acd13(66)
      acd13(66)=acd13(68)*acd13(66)
      acd13(68)=acd13(50)*acd13(63)
      acd13(70)=acd13(48)*acd13(62)
      acd13(71)=acd13(46)*acd13(61)
      acd13(72)=acd13(44)*acd13(60)
      acd13(73)=acd13(42)*acd13(59)
      acd13(74)=acd13(39)*acd13(58)
      acd13(75)=acd13(37)*acd13(57)
      acd13(76)=acd13(34)*acd13(56)
      acd13(77)=acd13(31)*acd13(55)
      acd13(78)=acd13(29)*acd13(54)
      acd13(79)=acd13(27)*acd13(53)
      acd13(80)=acd13(24)*acd13(51)
      acd13(81)=acd13(15)*acd13(52)
      acd13(82)=acd13(20)*acd13(40)
      acd13(83)=acd13(19)*acd13(35)
      acd13(84)=acd13(18)*acd13(32)
      acd13(85)=acd13(16)*acd13(21)
      acd13(86)=acd13(17)*acd13(22)
      acd13(66)=acd13(86)+acd13(85)+acd13(84)+acd13(83)+acd13(82)+acd13(81)+acd&
      &13(80)+acd13(79)+acd13(78)+acd13(77)+acd13(76)+acd13(75)+acd13(74)+acd13&
      &(73)+acd13(72)+acd13(71)+acd13(70)+acd13(64)+acd13(68)+acd13(69)+acd13(6&
      &6)
      brack(ninjaidxt1mu0)=acd13(67)
      brack(ninjaidxt0mu0)=acd13(66)
      brack(ninjaidxt0mu2)=-acd13(65)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d13h4_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd13h4
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k3+k4+k5
      vecA(1:4) = + a(0:3) - qshift(1:4)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d13h4l131
