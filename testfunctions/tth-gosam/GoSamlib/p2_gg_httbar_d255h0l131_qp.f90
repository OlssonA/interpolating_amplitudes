module     p2_gg_httbar_d255h0l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d255h0l131_qp.f90
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
      use p2_gg_httbar_abbrevd255h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd255
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd255h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(71) :: acd255
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd255(1)=dotproduct(ninjaE3,spval5e1)
      acd255(2)=dotproduct(ninjaE3,spvae1e2)
      acd255(3)=abb255(62)
      acd255(4)=dotproduct(ninjaE3,spval4e1)
      acd255(5)=abb255(35)
      acd255(6)=dotproduct(ninjaA,ninjaE3)
      acd255(7)=dotproduct(ninjaE3,spvae1k2)
      acd255(8)=dotproduct(ninjaE3,spval4e2)
      acd255(9)=dotproduct(ninjaE3,spvae2e1)
      acd255(10)=abb255(7)
      acd255(11)=dotproduct(ninjaE3,spval5e2)
      acd255(12)=abb255(28)
      acd255(13)=dotproduct(ninjaE3,spvae1l3)
      acd255(14)=abb255(60)
      acd255(15)=abb255(59)
      acd255(16)=dotproduct(ninjaE3,spvae2k2)
      acd255(17)=abb255(49)
      acd255(18)=dotproduct(k2,ninjaE3)
      acd255(19)=abb255(51)
      acd255(20)=abb255(39)
      acd255(21)=dotproduct(ninjaA,ninjaA)
      acd255(22)=dotproduct(ninjaA,spval5e1)
      acd255(23)=dotproduct(ninjaA,spvae1e2)
      acd255(24)=dotproduct(ninjaA,spval4e1)
      acd255(25)=abb255(26)
      acd255(26)=abb255(12)
      acd255(27)=abb255(19)
      acd255(28)=abb255(52)
      acd255(29)=abb255(18)
      acd255(30)=dotproduct(ninjaA,spvae1k2)
      acd255(31)=dotproduct(ninjaA,spval4e2)
      acd255(32)=dotproduct(ninjaA,spvae2e1)
      acd255(33)=dotproduct(ninjaA,spvae2k2)
      acd255(34)=dotproduct(ninjaA,spval5e2)
      acd255(35)=dotproduct(ninjaA,spvae1l3)
      acd255(36)=abb255(14)
      acd255(37)=dotproduct(ninjaE3,spvae2l3)
      acd255(38)=abb255(13)
      acd255(39)=abb255(21)
      acd255(40)=dotproduct(ninjaE3,spval4k2)
      acd255(41)=abb255(22)
      acd255(42)=abb255(24)
      acd255(43)=dotproduct(ninjaE3,spval3e2)
      acd255(44)=abb255(54)
      acd255(45)=abb255(57)
      acd255(46)=dotproduct(ninjaE3,spval5k1)
      acd255(47)=abb255(30)
      acd255(48)=dotproduct(ninjaE3,spval4k1)
      acd255(49)=abb255(38)
      acd255(50)=abb255(50)
      acd255(51)=abb255(53)
      acd255(52)=abb255(15)
      acd255(53)=abb255(58)
      acd255(54)=abb255(16)
      acd255(55)=abb255(34)
      acd255(56)=abb255(32)
      acd255(57)=acd255(4)*acd255(5)
      acd255(58)=acd255(1)*acd255(3)
      acd255(58)=acd255(57)+acd255(58)
      acd255(58)=acd255(2)*acd255(58)
      acd255(59)=2.0_ki*acd255(6)
      acd255(57)=acd255(59)*acd255(57)
      acd255(60)=acd255(16)*acd255(17)
      acd255(61)=acd255(3)*acd255(59)
      acd255(61)=-acd255(60)+acd255(61)
      acd255(61)=acd255(1)*acd255(61)
      acd255(57)=acd255(57)+acd255(61)
      acd255(57)=acd255(2)*acd255(57)
      acd255(61)=acd255(14)*acd255(8)
      acd255(62)=acd255(15)*acd255(11)
      acd255(61)=acd255(61)+acd255(62)
      acd255(61)=acd255(13)*acd255(61)
      acd255(62)=acd255(11)*acd255(12)
      acd255(63)=acd255(8)*acd255(10)
      acd255(62)=acd255(62)+acd255(63)
      acd255(63)=acd255(7)*acd255(62)
      acd255(63)=acd255(63)+acd255(61)
      acd255(63)=acd255(9)*acd255(63)
      acd255(57)=acd255(57)+acd255(63)
      acd255(63)=acd255(15)*acd255(34)
      acd255(64)=acd255(14)*acd255(31)
      acd255(63)=acd255(64)+acd255(50)+acd255(63)
      acd255(63)=acd255(13)*acd255(63)
      acd255(64)=acd255(15)*acd255(35)
      acd255(65)=acd255(12)*acd255(30)
      acd255(64)=acd255(65)+acd255(45)+acd255(64)
      acd255(64)=acd255(11)*acd255(64)
      acd255(65)=acd255(14)*acd255(35)
      acd255(66)=acd255(10)*acd255(30)
      acd255(65)=acd255(66)+acd255(42)+acd255(65)
      acd255(65)=acd255(8)*acd255(65)
      acd255(66)=acd255(12)*acd255(34)
      acd255(67)=acd255(10)*acd255(31)
      acd255(66)=acd255(67)+acd255(36)+acd255(66)
      acd255(66)=acd255(7)*acd255(66)
      acd255(67)=acd255(48)*acd255(49)
      acd255(68)=-acd255(46)*acd255(47)
      acd255(69)=acd255(43)*acd255(44)
      acd255(70)=acd255(18)*acd255(19)
      acd255(71)=acd255(26)*acd255(59)
      acd255(63)=acd255(71)+acd255(66)+acd255(65)+acd255(64)+acd255(63)+acd255(&
      &70)+acd255(69)+acd255(67)+acd255(68)
      acd255(63)=acd255(9)*acd255(63)
      acd255(64)=acd255(5)*acd255(24)
      acd255(65)=acd255(3)*acd255(22)
      acd255(64)=acd255(65)+acd255(28)+acd255(64)
      acd255(64)=acd255(64)*acd255(59)
      acd255(65)=-acd255(17)*acd255(33)
      acd255(66)=acd255(21)+ninjaP
      acd255(67)=acd255(3)*acd255(66)
      acd255(65)=acd255(67)+acd255(53)+acd255(65)
      acd255(65)=acd255(1)*acd255(65)
      acd255(67)=acd255(40)*acd255(56)
      acd255(68)=acd255(37)*acd255(51)
      acd255(69)=-acd255(17)*acd255(22)
      acd255(69)=acd255(54)+acd255(69)
      acd255(69)=acd255(16)*acd255(69)
      acd255(66)=acd255(5)*acd255(66)
      acd255(66)=acd255(55)+acd255(66)
      acd255(66)=acd255(4)*acd255(66)
      acd255(64)=acd255(65)+acd255(64)+acd255(66)+acd255(69)+acd255(67)+acd255(&
      &68)
      acd255(64)=acd255(2)*acd255(64)
      acd255(62)=acd255(32)*acd255(62)
      acd255(65)=acd255(40)*acd255(41)
      acd255(66)=acd255(37)*acd255(38)
      acd255(67)=acd255(16)*acd255(39)
      acd255(62)=acd255(67)+acd255(65)+acd255(66)+acd255(62)
      acd255(62)=acd255(7)*acd255(62)
      acd255(65)=acd255(3)*acd255(23)
      acd255(65)=acd255(27)+acd255(65)
      acd255(65)=acd255(65)*acd255(59)
      acd255(66)=acd255(43)*acd255(52)
      acd255(67)=acd255(18)*acd255(20)
      acd255(60)=-acd255(23)*acd255(60)
      acd255(60)=acd255(65)+acd255(60)+acd255(66)+acd255(67)
      acd255(60)=acd255(1)*acd255(60)
      acd255(61)=acd255(32)*acd255(61)
      acd255(65)=acd255(5)*acd255(23)
      acd255(65)=acd255(29)+acd255(65)
      acd255(65)=acd255(4)*acd255(65)
      acd255(66)=acd255(7)*acd255(25)
      acd255(65)=acd255(65)+acd255(66)
      acd255(59)=acd255(65)*acd255(59)
      acd255(59)=acd255(63)+acd255(64)+acd255(60)+acd255(59)+acd255(62)+acd255(&
      &61)
      brack(ninjaidxt1mu0)=acd255(57)
      brack(ninjaidxt0mu0)=acd255(59)
      brack(ninjaidxt0mu2)=acd255(58)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d255h0_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd255h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k3-k5
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
end module     p2_gg_httbar_d255h0l131_qp
