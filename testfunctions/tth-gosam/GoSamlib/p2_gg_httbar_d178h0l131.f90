module     p2_gg_httbar_d178h0l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d178h0l131.f90
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
      use p2_gg_httbar_abbrevd178h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(28) :: acd178
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd178(1)=dotproduct(k2,ninjaE3)
      acd178(2)=dotproduct(e1,ninjaE3)
      acd178(3)=abb178(62)
      acd178(4)=dotproduct(l4,ninjaE3)
      acd178(5)=abb178(45)
      acd178(6)=dotproduct(ninjaE3,spvak1k2)
      acd178(7)=abb178(17)
      acd178(8)=dotproduct(ninjaE3,spval4k2)
      acd178(9)=abb178(23)
      acd178(10)=dotproduct(ninjaE3,spval4k1)
      acd178(11)=abb178(26)
      acd178(12)=dotproduct(ninjaE3,spval4l5)
      acd178(13)=abb178(27)
      acd178(14)=dotproduct(ninjaE3,spval5k2)
      acd178(15)=abb178(29)
      acd178(16)=dotproduct(ninjaE3,spvae2k2)
      acd178(17)=abb178(49)
      acd178(18)=dotproduct(ninjaE3,spval4e2)
      acd178(19)=abb178(84)
      acd178(20)=acd178(3)*acd178(1)
      acd178(21)=acd178(5)*acd178(4)
      acd178(22)=acd178(7)*acd178(6)
      acd178(23)=acd178(9)*acd178(8)
      acd178(24)=acd178(11)*acd178(10)
      acd178(25)=acd178(13)*acd178(12)
      acd178(26)=acd178(15)*acd178(14)
      acd178(27)=acd178(17)*acd178(16)
      acd178(28)=acd178(19)*acd178(18)
      acd178(20)=acd178(28)+acd178(27)+acd178(26)+acd178(25)+acd178(24)+acd178(&
      &23)+acd178(22)+acd178(20)+acd178(21)
      acd178(20)=acd178(2)*acd178(20)
      brack(ninjaidxt2mu0)=acd178(20)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd178h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(129) :: acd178
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd178(1)=dotproduct(k2,ninjaE3)
      acd178(2)=dotproduct(e1,ninjaE4)
      acd178(3)=abb178(62)
      acd178(4)=dotproduct(k2,ninjaE4)
      acd178(5)=dotproduct(e1,ninjaE3)
      acd178(6)=dotproduct(l4,ninjaE3)
      acd178(7)=abb178(45)
      acd178(8)=dotproduct(l4,ninjaE4)
      acd178(9)=dotproduct(ninjaE4,spvak1k2)
      acd178(10)=abb178(17)
      acd178(11)=dotproduct(ninjaE4,spval4k2)
      acd178(12)=abb178(23)
      acd178(13)=dotproduct(ninjaE4,spval4k1)
      acd178(14)=abb178(26)
      acd178(15)=dotproduct(ninjaE4,spval4l5)
      acd178(16)=abb178(27)
      acd178(17)=dotproduct(ninjaE4,spval5k2)
      acd178(18)=abb178(29)
      acd178(19)=dotproduct(ninjaE4,spvae2k2)
      acd178(20)=abb178(49)
      acd178(21)=dotproduct(ninjaE4,spval4e2)
      acd178(22)=abb178(84)
      acd178(23)=dotproduct(ninjaE3,spvak1k2)
      acd178(24)=dotproduct(ninjaE3,spval4k2)
      acd178(25)=dotproduct(ninjaE3,spval4k1)
      acd178(26)=dotproduct(ninjaE3,spval4l5)
      acd178(27)=dotproduct(ninjaE3,spval5k2)
      acd178(28)=dotproduct(ninjaE3,spvae2k2)
      acd178(29)=dotproduct(ninjaE3,spval4e2)
      acd178(30)=abb178(83)
      acd178(31)=dotproduct(k2,ninjaA)
      acd178(32)=dotproduct(e1,ninjaA)
      acd178(33)=abb178(50)
      acd178(34)=dotproduct(l4,ninjaA)
      acd178(35)=abb178(18)
      acd178(36)=dotproduct(ninjaA,spvak1k2)
      acd178(37)=dotproduct(ninjaA,spval4k2)
      acd178(38)=dotproduct(ninjaA,spval4k1)
      acd178(39)=dotproduct(ninjaA,spval4l5)
      acd178(40)=dotproduct(ninjaA,spval5k2)
      acd178(41)=dotproduct(ninjaA,spvae2k2)
      acd178(42)=dotproduct(ninjaA,spval4e2)
      acd178(43)=abb178(15)
      acd178(44)=dotproduct(ninjaA,ninjaE3)
      acd178(45)=dotproduct(ninjaE3,spvak2l4)
      acd178(46)=abb178(11)
      acd178(47)=dotproduct(ninjaE3,spvak2e2)
      acd178(48)=abb178(12)
      acd178(49)=dotproduct(ninjaE3,spvae1e2)
      acd178(50)=abb178(14)
      acd178(51)=dotproduct(ninjaE3,spvae1k2)
      acd178(52)=abb178(16)
      acd178(53)=dotproduct(ninjaE3,spvak2k1)
      acd178(54)=abb178(19)
      acd178(55)=abb178(20)
      acd178(56)=dotproduct(ninjaE3,spvak1l4)
      acd178(57)=abb178(21)
      acd178(58)=dotproduct(ninjaE3,spvak1e1)
      acd178(59)=abb178(22)
      acd178(60)=dotproduct(ninjaE3,spval5l4)
      acd178(61)=abb178(24)
      acd178(62)=abb178(25)
      acd178(63)=abb178(37)
      acd178(64)=dotproduct(ninjaE3,spvae1k1)
      acd178(65)=abb178(28)
      acd178(66)=dotproduct(ninjaE3,spvae2l4)
      acd178(67)=abb178(31)
      acd178(68)=dotproduct(ninjaE3,spval4e1)
      acd178(69)=abb178(33)
      acd178(70)=dotproduct(ninjaE3,spvak2e1)
      acd178(71)=abb178(48)
      acd178(72)=dotproduct(ninjaE3,spvae2e1)
      acd178(73)=abb178(52)
      acd178(74)=abb178(54)
      acd178(75)=dotproduct(ninjaE3,spval5e1)
      acd178(76)=abb178(67)
      acd178(77)=dotproduct(ninjaE3,spvae1l4)
      acd178(78)=abb178(85)
      acd178(79)=dotproduct(ninjaE3,spvae1l5)
      acd178(80)=abb178(154)
      acd178(81)=dotproduct(ninjaE3,spvak2l5)
      acd178(82)=abb178(166)
      acd178(83)=dotproduct(ninjaA,ninjaA)
      acd178(84)=dotproduct(ninjaA,spvak2l4)
      acd178(85)=dotproduct(ninjaA,spvak2e2)
      acd178(86)=dotproduct(ninjaA,spvae1e2)
      acd178(87)=dotproduct(ninjaA,spvae1k2)
      acd178(88)=dotproduct(ninjaA,spvak2k1)
      acd178(89)=dotproduct(ninjaA,spvak1l4)
      acd178(90)=dotproduct(ninjaA,spvak1e1)
      acd178(91)=dotproduct(ninjaA,spval5l4)
      acd178(92)=dotproduct(ninjaA,spvae1k1)
      acd178(93)=dotproduct(ninjaA,spvae2l4)
      acd178(94)=dotproduct(ninjaA,spval4e1)
      acd178(95)=dotproduct(ninjaA,spvak2e1)
      acd178(96)=dotproduct(ninjaA,spvae2e1)
      acd178(97)=dotproduct(ninjaA,spval5e1)
      acd178(98)=dotproduct(ninjaA,spvae1l4)
      acd178(99)=dotproduct(ninjaA,spvae1l5)
      acd178(100)=dotproduct(ninjaA,spvak2l5)
      acd178(101)=abb178(13)
      acd178(102)=acd178(22)*acd178(21)
      acd178(103)=acd178(20)*acd178(19)
      acd178(104)=acd178(18)*acd178(17)
      acd178(105)=acd178(16)*acd178(15)
      acd178(106)=acd178(14)*acd178(13)
      acd178(107)=acd178(12)*acd178(11)
      acd178(108)=acd178(10)*acd178(9)
      acd178(109)=acd178(7)*acd178(8)
      acd178(110)=acd178(3)*acd178(4)
      acd178(102)=acd178(106)+acd178(107)+acd178(108)+acd178(109)+acd178(102)+a&
      &cd178(104)+acd178(105)+acd178(103)+acd178(110)
      acd178(102)=acd178(102)*acd178(5)
      acd178(103)=acd178(22)*acd178(29)
      acd178(104)=acd178(20)*acd178(28)
      acd178(105)=acd178(18)*acd178(27)
      acd178(106)=acd178(16)*acd178(26)
      acd178(107)=acd178(14)*acd178(25)
      acd178(108)=acd178(12)*acd178(24)
      acd178(109)=acd178(10)*acd178(23)
      acd178(110)=acd178(7)*acd178(6)
      acd178(111)=acd178(3)*acd178(1)
      acd178(103)=acd178(107)+acd178(106)+acd178(105)+acd178(103)+acd178(104)+a&
      &cd178(108)+acd178(109)+acd178(110)+acd178(111)
      acd178(104)=acd178(103)*acd178(2)
      acd178(102)=acd178(102)+acd178(104)-acd178(30)
      acd178(103)=acd178(32)*acd178(103)
      acd178(104)=acd178(22)*acd178(42)
      acd178(105)=acd178(20)*acd178(41)
      acd178(106)=acd178(18)*acd178(40)
      acd178(107)=acd178(16)*acd178(39)
      acd178(108)=acd178(14)*acd178(38)
      acd178(109)=acd178(12)*acd178(37)
      acd178(110)=acd178(10)*acd178(36)
      acd178(111)=acd178(7)*acd178(34)
      acd178(112)=acd178(3)*acd178(31)
      acd178(104)=acd178(110)+acd178(111)+acd178(112)+acd178(43)+acd178(106)+ac&
      &d178(107)+acd178(108)+acd178(109)+acd178(104)+acd178(105)
      acd178(105)=acd178(5)*acd178(104)
      acd178(106)=-acd178(82)*acd178(81)
      acd178(107)=acd178(80)*acd178(79)
      acd178(108)=acd178(78)*acd178(77)
      acd178(109)=acd178(76)*acd178(75)
      acd178(110)=acd178(73)*acd178(72)
      acd178(111)=acd178(71)*acd178(70)
      acd178(112)=acd178(69)*acd178(68)
      acd178(113)=acd178(67)*acd178(66)
      acd178(114)=acd178(65)*acd178(64)
      acd178(115)=acd178(61)*acd178(60)
      acd178(116)=acd178(59)*acd178(58)
      acd178(117)=acd178(57)*acd178(56)
      acd178(118)=acd178(54)*acd178(53)
      acd178(119)=acd178(52)*acd178(51)
      acd178(120)=acd178(50)*acd178(49)
      acd178(121)=acd178(48)*acd178(47)
      acd178(122)=acd178(46)*acd178(45)
      acd178(123)=acd178(30)*acd178(44)
      acd178(124)=acd178(29)*acd178(74)
      acd178(125)=acd178(26)*acd178(63)
      acd178(126)=acd178(25)*acd178(62)
      acd178(127)=acd178(24)*acd178(55)
      acd178(128)=acd178(6)*acd178(35)
      acd178(129)=acd178(1)*acd178(33)
      acd178(103)=acd178(105)+acd178(103)+acd178(129)+acd178(128)+acd178(127)+a&
      &cd178(126)+acd178(125)+acd178(124)-2.0_ki*acd178(123)+acd178(122)+acd178&
      &(121)+acd178(120)+acd178(119)+acd178(118)+acd178(117)+acd178(116)+acd178&
      &(115)+acd178(114)+acd178(113)+acd178(112)+acd178(111)+acd178(110)+acd178&
      &(109)+acd178(108)+acd178(106)+acd178(107)
      acd178(105)=ninjaP*acd178(102)
      acd178(104)=acd178(32)*acd178(104)
      acd178(106)=-acd178(82)*acd178(100)
      acd178(107)=acd178(80)*acd178(99)
      acd178(108)=acd178(78)*acd178(98)
      acd178(109)=acd178(76)*acd178(97)
      acd178(110)=acd178(73)*acd178(96)
      acd178(111)=acd178(71)*acd178(95)
      acd178(112)=acd178(69)*acd178(94)
      acd178(113)=acd178(67)*acd178(93)
      acd178(114)=acd178(65)*acd178(92)
      acd178(115)=acd178(61)*acd178(91)
      acd178(116)=acd178(59)*acd178(90)
      acd178(117)=acd178(57)*acd178(89)
      acd178(118)=acd178(54)*acd178(88)
      acd178(119)=acd178(52)*acd178(87)
      acd178(120)=acd178(50)*acd178(86)
      acd178(121)=acd178(48)*acd178(85)
      acd178(122)=acd178(46)*acd178(84)
      acd178(123)=acd178(42)*acd178(74)
      acd178(124)=acd178(39)*acd178(63)
      acd178(125)=acd178(38)*acd178(62)
      acd178(126)=acd178(37)*acd178(55)
      acd178(127)=acd178(34)*acd178(35)
      acd178(128)=acd178(31)*acd178(33)
      acd178(129)=-acd178(30)*acd178(83)
      acd178(104)=acd178(104)+acd178(129)+acd178(128)+acd178(127)+acd178(126)+a&
      &cd178(125)+acd178(124)+acd178(123)+acd178(122)+acd178(121)+acd178(120)+a&
      &cd178(119)+acd178(118)+acd178(117)+acd178(116)+acd178(115)+acd178(114)+a&
      &cd178(113)+acd178(112)+acd178(111)+acd178(110)+acd178(109)+acd178(108)+a&
      &cd178(107)+acd178(101)+acd178(106)+acd178(105)
      brack(ninjaidxt1mu0)=acd178(103)
      brack(ninjaidxt0mu0)=acd178(104)
      brack(ninjaidxt0mu2)=acd178(102)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d178h0_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd178h0
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k3-k4-k5
      vecA(1:4) = - a(0:3) - qshift(1:4)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d178h0l131
