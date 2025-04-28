module     p2_gg_httbar_d86h0l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d86h0l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd86h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(13) :: acd86
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd86(1)=dotproduct(e1,ninjaE3)
      acd86(2)=dotproduct(ninjaE3,spvae2k2)
      acd86(3)=dotproduct(ninjaE3,spval5e2)
      acd86(4)=abb86(8)
      acd86(5)=dotproduct(ninjaE3,spval4e2)
      acd86(6)=abb86(12)
      acd86(7)=dotproduct(ninjaE3,spval3e2)
      acd86(8)=abb86(31)
      acd86(9)=dotproduct(ninjaE3,spvae2l3)
      acd86(10)=abb86(68)
      acd86(11)=acd86(4)*acd86(3)
      acd86(12)=acd86(6)*acd86(5)
      acd86(13)=acd86(8)*acd86(7)
      acd86(11)=acd86(13)+acd86(11)+acd86(12)
      acd86(11)=acd86(11)*acd86(2)
      acd86(12)=acd86(10)*acd86(9)*acd86(3)
      acd86(11)=acd86(12)+acd86(11)
      acd86(11)=acd86(1)*acd86(11)
      brack(ninjaidxt2mu0)=acd86(11)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd86h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(119) :: acd86
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd86(1)=dotproduct(e1,ninjaE3)
      acd86(2)=dotproduct(ninjaE3,spvae2k2)
      acd86(3)=dotproduct(ninjaE4,spval5e2)
      acd86(4)=abb86(8)
      acd86(5)=dotproduct(ninjaE4,spval4e2)
      acd86(6)=abb86(12)
      acd86(7)=dotproduct(ninjaE4,spval3e2)
      acd86(8)=abb86(31)
      acd86(9)=dotproduct(ninjaE3,spval5e2)
      acd86(10)=dotproduct(ninjaE4,spvae2k2)
      acd86(11)=dotproduct(ninjaE4,spvae2l3)
      acd86(12)=abb86(68)
      acd86(13)=dotproduct(ninjaE3,spval4e2)
      acd86(14)=dotproduct(ninjaE3,spvae2l3)
      acd86(15)=dotproduct(ninjaE3,spval3e2)
      acd86(16)=dotproduct(e1,ninjaE4)
      acd86(17)=dotproduct(ninjaE3,spvae2e1)
      acd86(18)=abb86(44)
      acd86(19)=dotproduct(ninjaE3,spvae1e2)
      acd86(20)=abb86(45)
      acd86(21)=dotproduct(k2,ninjaE3)
      acd86(22)=abb86(25)
      acd86(23)=dotproduct(l5,ninjaE3)
      acd86(24)=abb86(48)
      acd86(25)=dotproduct(e1,ninjaA)
      acd86(26)=dotproduct(ninjaA,spvae2k2)
      acd86(27)=dotproduct(ninjaA,spval5e2)
      acd86(28)=dotproduct(ninjaA,spval4e2)
      acd86(29)=dotproduct(ninjaA,spvae2l3)
      acd86(30)=dotproduct(ninjaA,spval3e2)
      acd86(31)=abb86(11)
      acd86(32)=abb86(17)
      acd86(33)=abb86(9)
      acd86(34)=abb86(10)
      acd86(35)=abb86(70)
      acd86(36)=dotproduct(ninjaA,ninjaE3)
      acd86(37)=abb86(24)
      acd86(38)=abb86(36)
      acd86(39)=abb86(55)
      acd86(40)=abb86(63)
      acd86(41)=dotproduct(ninjaE3,spval5k2)
      acd86(42)=abb86(14)
      acd86(43)=abb86(23)
      acd86(44)=dotproduct(ninjaE3,spvak1e2)
      acd86(45)=abb86(15)
      acd86(46)=dotproduct(ninjaE3,spval5l3)
      acd86(47)=abb86(39)
      acd86(48)=abb86(73)
      acd86(49)=dotproduct(ninjaE3,spvak2l3)
      acd86(50)=abb86(46)
      acd86(51)=dotproduct(ninjaE3,spval3k2)
      acd86(52)=abb86(32)
      acd86(53)=dotproduct(ninjaE3,spval4l5)
      acd86(54)=abb86(29)
      acd86(55)=dotproduct(ninjaE3,spval4k2)
      acd86(56)=abb86(33)
      acd86(57)=dotproduct(ninjaE3,spvae2k1)
      acd86(58)=abb86(34)
      acd86(59)=dotproduct(ninjaE3,spval3l5)
      acd86(60)=abb86(38)
      acd86(61)=dotproduct(k2,ninjaA)
      acd86(62)=dotproduct(ninjaA,spvae2e1)
      acd86(63)=abb86(51)
      acd86(64)=dotproduct(l5,ninjaA)
      acd86(65)=dotproduct(ninjaA,spvae1e2)
      acd86(66)=abb86(40)
      acd86(67)=abb86(21)
      acd86(68)=dotproduct(ninjaA,ninjaA)
      acd86(69)=abb86(22)
      acd86(70)=dotproduct(ninjaA,spval5k2)
      acd86(71)=dotproduct(ninjaA,spvak1e2)
      acd86(72)=dotproduct(ninjaA,spval3k2)
      acd86(73)=dotproduct(ninjaA,spval4l5)
      acd86(74)=dotproduct(ninjaA,spval5l3)
      acd86(75)=dotproduct(ninjaA,spval4k2)
      acd86(76)=dotproduct(ninjaA,spvae2k1)
      acd86(77)=dotproduct(ninjaA,spval3l5)
      acd86(78)=dotproduct(ninjaA,spvak2l3)
      acd86(79)=abb86(18)
      acd86(80)=abb86(27)
      acd86(81)=abb86(53)
      acd86(82)=abb86(58)
      acd86(83)=abb86(19)
      acd86(84)=abb86(43)
      acd86(85)=abb86(16)
      acd86(86)=abb86(20)
      acd86(87)=abb86(28)
      acd86(88)=abb86(26)
      acd86(89)=abb86(30)
      acd86(90)=abb86(69)
      acd86(91)=abb86(37)
      acd86(92)=abb86(42)
      acd86(93)=abb86(35)
      acd86(94)=abb86(41)
      acd86(95)=acd86(8)*acd86(7)
      acd86(96)=acd86(6)*acd86(5)
      acd86(97)=acd86(4)*acd86(3)
      acd86(95)=acd86(97)+acd86(95)+acd86(96)
      acd86(95)=acd86(95)*acd86(2)
      acd86(96)=acd86(12)*acd86(11)
      acd86(97)=acd86(4)*acd86(10)
      acd86(96)=acd86(96)+acd86(97)
      acd86(96)=acd86(96)*acd86(9)
      acd86(97)=acd86(8)*acd86(15)
      acd86(98)=acd86(6)*acd86(13)
      acd86(97)=acd86(97)+acd86(98)
      acd86(98)=acd86(10)*acd86(97)
      acd86(99)=acd86(12)*acd86(14)
      acd86(100)=acd86(99)*acd86(3)
      acd86(95)=acd86(95)+acd86(96)+acd86(98)+acd86(100)
      acd86(96)=acd86(1)*acd86(95)
      acd86(98)=acd86(9)*acd86(4)
      acd86(98)=acd86(97)+acd86(98)
      acd86(100)=acd86(98)*acd86(16)
      acd86(101)=acd86(2)*acd86(100)
      acd86(102)=acd86(99)*acd86(9)
      acd86(103)=acd86(16)*acd86(102)
      acd86(104)=acd86(17)*acd86(18)
      acd86(105)=acd86(19)*acd86(20)
      acd86(96)=acd86(96)+acd86(101)+acd86(105)+acd86(103)+acd86(104)
      acd86(101)=acd86(59)*acd86(60)
      acd86(103)=acd86(57)*acd86(58)
      acd86(104)=acd86(55)*acd86(56)
      acd86(105)=acd86(53)*acd86(54)
      acd86(106)=acd86(51)*acd86(52)
      acd86(107)=acd86(23)*acd86(24)
      acd86(108)=acd86(41)*acd86(43)
      acd86(109)=2.0_ki*acd86(36)
      acd86(110)=acd86(109)*acd86(20)
      acd86(101)=-acd86(103)-acd86(104)-acd86(105)-acd86(106)-acd86(110)-acd86(&
      &101)+acd86(107)-acd86(108)
      acd86(103)=acd86(14)*acd86(40)
      acd86(103)=acd86(103)-acd86(101)
      acd86(103)=acd86(19)*acd86(103)
      acd86(104)=acd86(49)*acd86(50)
      acd86(105)=acd86(46)*acd86(47)
      acd86(106)=acd86(44)*acd86(45)
      acd86(107)=acd86(21)*acd86(22)
      acd86(108)=acd86(41)*acd86(42)
      acd86(110)=acd86(109)*acd86(18)
      acd86(104)=acd86(104)+acd86(105)+acd86(106)+acd86(107)+acd86(108)+acd86(1&
      &10)
      acd86(105)=acd86(15)*acd86(48)
      acd86(106)=acd86(13)*acd86(39)
      acd86(107)=acd86(9)*acd86(38)
      acd86(105)=acd86(107)+acd86(106)+acd86(105)+acd86(104)
      acd86(105)=acd86(17)*acd86(105)
      acd86(97)=acd86(26)*acd86(97)
      acd86(106)=acd86(15)*acd86(35)
      acd86(107)=acd86(14)*acd86(34)
      acd86(108)=acd86(13)*acd86(33)
      acd86(110)=acd86(99)*acd86(27)
      acd86(97)=acd86(106)+acd86(107)+acd86(108)+acd86(110)+acd86(97)
      acd86(106)=acd86(8)*acd86(30)
      acd86(106)=acd86(106)+acd86(31)
      acd86(107)=acd86(6)*acd86(28)
      acd86(108)=acd86(4)*acd86(27)
      acd86(107)=acd86(108)+acd86(106)+acd86(107)
      acd86(108)=acd86(2)*acd86(107)
      acd86(110)=acd86(12)*acd86(29)
      acd86(111)=acd86(4)*acd86(26)
      acd86(110)=acd86(110)+acd86(111)+acd86(32)
      acd86(111)=acd86(9)*acd86(110)
      acd86(108)=acd86(108)+acd86(111)+acd86(97)
      acd86(108)=acd86(1)*acd86(108)
      acd86(98)=acd86(25)*acd86(98)
      acd86(111)=acd86(19)*acd86(37)
      acd86(98)=acd86(111)+acd86(98)
      acd86(98)=acd86(2)*acd86(98)
      acd86(102)=acd86(25)*acd86(102)
      acd86(98)=acd86(108)+acd86(98)+acd86(103)+acd86(102)+acd86(105)
      acd86(102)=ninjaP+acd86(68)
      acd86(103)=acd86(20)*acd86(102)
      acd86(105)=acd86(60)*acd86(77)
      acd86(108)=acd86(58)*acd86(76)
      acd86(111)=acd86(56)*acd86(75)
      acd86(112)=acd86(54)*acd86(73)
      acd86(113)=acd86(52)*acd86(72)
      acd86(114)=acd86(43)*acd86(70)
      acd86(115)=-acd86(24)*acd86(64)
      acd86(116)=acd86(29)*acd86(40)
      acd86(117)=acd86(26)*acd86(37)
      acd86(103)=acd86(117)+acd86(116)+acd86(115)+acd86(114)+acd86(113)+acd86(1&
      &12)+acd86(111)+acd86(108)+acd86(87)+acd86(105)+acd86(103)
      acd86(103)=acd86(19)*acd86(103)
      acd86(102)=acd86(18)*acd86(102)
      acd86(105)=acd86(50)*acd86(78)
      acd86(108)=acd86(47)*acd86(74)
      acd86(111)=acd86(45)*acd86(71)
      acd86(112)=acd86(42)*acd86(70)
      acd86(113)=acd86(22)*acd86(61)
      acd86(114)=acd86(30)*acd86(48)
      acd86(115)=acd86(28)*acd86(39)
      acd86(116)=acd86(27)*acd86(38)
      acd86(102)=acd86(116)+acd86(115)+acd86(114)+acd86(113)+acd86(112)+acd86(1&
      &11)+acd86(108)+acd86(84)+acd86(105)+acd86(102)
      acd86(102)=acd86(17)*acd86(102)
      acd86(101)=-acd86(65)*acd86(101)
      acd86(104)=acd86(62)*acd86(104)
      acd86(95)=ninjaP*acd86(95)
      acd86(105)=acd86(27)*acd86(110)
      acd86(108)=acd86(6)*acd86(26)
      acd86(108)=acd86(108)+acd86(33)
      acd86(108)=acd86(28)*acd86(108)
      acd86(106)=acd86(26)*acd86(106)
      acd86(111)=acd86(30)*acd86(35)
      acd86(112)=acd86(29)*acd86(34)
      acd86(95)=acd86(112)+acd86(67)+acd86(111)+acd86(95)+acd86(105)+acd86(106)&
      &+acd86(108)
      acd86(95)=acd86(1)*acd86(95)
      acd86(97)=acd86(25)*acd86(97)
      acd86(105)=acd86(25)*acd86(110)
      acd86(106)=acd86(62)*acd86(38)
      acd86(99)=ninjaP*acd86(99)*acd86(16)
      acd86(99)=acd86(105)+acd86(99)+acd86(80)+acd86(106)
      acd86(99)=acd86(9)*acd86(99)
      acd86(105)=acd86(25)*acd86(107)
      acd86(100)=ninjaP*acd86(100)
      acd86(106)=acd86(65)*acd86(37)
      acd86(100)=acd86(105)+acd86(79)+acd86(106)+acd86(100)
      acd86(100)=acd86(2)*acd86(100)
      acd86(105)=acd86(59)*acd86(93)
      acd86(106)=acd86(57)*acd86(92)
      acd86(107)=acd86(55)*acd86(91)
      acd86(108)=acd86(53)*acd86(88)
      acd86(110)=acd86(51)*acd86(86)
      acd86(111)=acd86(49)*acd86(94)
      acd86(112)=acd86(46)*acd86(89)
      acd86(113)=acd86(44)*acd86(85)
      acd86(114)=acd86(23)*acd86(66)
      acd86(115)=acd86(21)*acd86(63)
      acd86(116)=acd86(41)*acd86(83)
      acd86(109)=acd86(69)*acd86(109)
      acd86(117)=acd86(62)*acd86(48)
      acd86(117)=acd86(90)+acd86(117)
      acd86(117)=acd86(15)*acd86(117)
      acd86(118)=acd86(65)*acd86(40)
      acd86(118)=acd86(82)+acd86(118)
      acd86(118)=acd86(14)*acd86(118)
      acd86(119)=acd86(62)*acd86(39)
      acd86(119)=acd86(81)+acd86(119)
      acd86(119)=acd86(13)*acd86(119)
      acd86(95)=acd86(95)+acd86(100)+acd86(103)+acd86(102)+acd86(99)+acd86(97)+&
      &acd86(119)+acd86(118)+acd86(117)+acd86(101)+acd86(104)+acd86(109)+acd86(&
      &116)+acd86(115)+acd86(114)+acd86(113)+acd86(112)+acd86(111)+acd86(110)+a&
      &cd86(108)+acd86(107)+acd86(105)+acd86(106)
      brack(ninjaidxt1mu0)=acd86(98)
      brack(ninjaidxt0mu0)=acd86(95)
      brack(ninjaidxt0mu2)=acd86(96)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d86h0_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd86h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k2+k3+k4+k5
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
end module     p2_gg_httbar_d86h0l131_qp
