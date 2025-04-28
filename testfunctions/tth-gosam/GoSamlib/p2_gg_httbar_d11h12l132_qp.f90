module     p2_gg_httbar_d11h12l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d11h12l132_qp.f90
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
      use p2_gg_httbar_abbrevd11h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(14) :: acd11
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd11(1)=dotproduct(k1,ninjaE3)
      acd11(2)=dotproduct(ninjaE3,spvak2l5)
      acd11(3)=abb11(24)
      acd11(4)=dotproduct(ninjaE3,spval3l5)
      acd11(5)=abb11(34)
      acd11(6)=dotproduct(ninjaE3,spvak2l4)
      acd11(7)=abb11(40)
      acd11(8)=dotproduct(ninjaE3,spvak2l3)
      acd11(9)=abb11(47)
      acd11(10)=dotproduct(k2,ninjaE3)
      acd11(11)=-acd11(3)*acd11(2)
      acd11(12)=-acd11(5)*acd11(4)
      acd11(13)=acd11(7)*acd11(6)
      acd11(14)=acd11(9)*acd11(8)
      acd11(11)=acd11(14)+acd11(13)+acd11(11)+acd11(12)
      acd11(12)=acd11(10)-acd11(1)
      acd11(11)=acd11(12)*acd11(11)
      brack(ninjaidxt1x0mu0)=acd11(11)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd11h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(70) :: acd11
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd11(1)=dotproduct(k1,ninjaA1)
      acd11(2)=dotproduct(ninjaE3,spvak2l5)
      acd11(3)=abb11(24)
      acd11(4)=dotproduct(ninjaE3,spvak2l4)
      acd11(5)=abb11(40)
      acd11(6)=dotproduct(ninjaE3,spvak2l3)
      acd11(7)=abb11(47)
      acd11(8)=dotproduct(ninjaE3,spval3l5)
      acd11(9)=abb11(34)
      acd11(10)=dotproduct(k1,ninjaE3)
      acd11(11)=dotproduct(ninjaA1,spvak2l5)
      acd11(12)=dotproduct(ninjaA1,spvak2l4)
      acd11(13)=dotproduct(ninjaA1,spvak2l3)
      acd11(14)=dotproduct(ninjaA1,spval3l5)
      acd11(15)=dotproduct(k2,ninjaA1)
      acd11(16)=dotproduct(k2,ninjaE3)
      acd11(17)=dotproduct(k1,ninjaA0)
      acd11(18)=dotproduct(ninjaA0,spvak2l5)
      acd11(19)=dotproduct(ninjaA0,spvak2l4)
      acd11(20)=dotproduct(ninjaA0,spvak2l3)
      acd11(21)=dotproduct(ninjaA0,spval3l5)
      acd11(22)=abb11(22)
      acd11(23)=dotproduct(k2,ninjaA0)
      acd11(24)=abb11(13)
      acd11(25)=dotproduct(l5,ninjaE3)
      acd11(26)=abb11(15)
      acd11(27)=dotproduct(ninjaA0,ninjaE3)
      acd11(28)=abb11(23)
      acd11(29)=dotproduct(ninjaE3,spvak2k1)
      acd11(30)=abb11(10)
      acd11(31)=dotproduct(ninjaE3,spvak1l4)
      acd11(32)=abb11(11)
      acd11(33)=dotproduct(ninjaE3,spvak1l3)
      acd11(34)=abb11(12)
      acd11(35)=dotproduct(ninjaE3,spval3k1)
      acd11(36)=abb11(14)
      acd11(37)=abb11(16)
      acd11(38)=dotproduct(ninjaE3,spval3k2)
      acd11(39)=abb11(17)
      acd11(40)=dotproduct(ninjaE3,spvak1l5)
      acd11(41)=abb11(18)
      acd11(42)=abb11(19)
      acd11(43)=abb11(20)
      acd11(44)=dotproduct(ninjaE3,spvak1k2)
      acd11(45)=abb11(21)
      acd11(46)=abb11(37)
      acd11(47)=dotproduct(ninjaE3,spval5l4)
      acd11(48)=abb11(36)
      acd11(49)=dotproduct(ninjaE3,spval5l3)
      acd11(50)=abb11(38)
      acd11(51)=acd11(2)*acd11(3)
      acd11(52)=acd11(4)*acd11(5)
      acd11(53)=acd11(6)*acd11(7)
      acd11(54)=acd11(8)*acd11(9)
      acd11(51)=-acd11(51)+acd11(52)+acd11(53)-acd11(54)
      acd11(52)=-acd11(1)+acd11(15)
      acd11(52)=acd11(51)*acd11(52)
      acd11(53)=-acd11(11)*acd11(3)
      acd11(54)=acd11(12)*acd11(5)
      acd11(55)=acd11(13)*acd11(7)
      acd11(56)=-acd11(14)*acd11(9)
      acd11(53)=acd11(56)+acd11(55)+acd11(54)+acd11(53)
      acd11(54)=acd11(16)-acd11(10)
      acd11(53)=acd11(54)*acd11(53)
      acd11(52)=acd11(53)+acd11(52)
      acd11(53)=-acd11(17)+acd11(23)
      acd11(51)=acd11(51)*acd11(53)
      acd11(53)=-acd11(18)*acd11(3)
      acd11(55)=acd11(19)*acd11(5)
      acd11(56)=acd11(20)*acd11(7)
      acd11(57)=-acd11(21)*acd11(9)
      acd11(53)=acd11(57)+acd11(56)+acd11(55)+acd11(53)
      acd11(53)=acd11(54)*acd11(53)
      acd11(54)=acd11(22)*acd11(10)
      acd11(55)=acd11(24)*acd11(16)
      acd11(56)=acd11(26)*acd11(25)
      acd11(57)=acd11(28)*acd11(27)
      acd11(58)=acd11(30)*acd11(29)
      acd11(59)=acd11(32)*acd11(31)
      acd11(60)=acd11(34)*acd11(33)
      acd11(61)=acd11(36)*acd11(35)
      acd11(62)=acd11(37)*acd11(2)
      acd11(63)=acd11(39)*acd11(38)
      acd11(64)=acd11(41)*acd11(40)
      acd11(65)=acd11(42)*acd11(4)
      acd11(66)=acd11(43)*acd11(6)
      acd11(67)=acd11(45)*acd11(44)
      acd11(68)=acd11(46)*acd11(8)
      acd11(69)=acd11(48)*acd11(47)
      acd11(70)=acd11(50)*acd11(49)
      acd11(51)=acd11(70)+acd11(69)+acd11(68)+acd11(67)+acd11(66)+acd11(65)+acd&
      &11(64)+acd11(63)+acd11(62)+acd11(61)+acd11(60)+acd11(59)+acd11(58)+2.0_k&
      &i*acd11(57)+acd11(56)+acd11(55)+acd11(54)+acd11(53)+acd11(51)
      brack(ninjaidxt0x0mu0)=acd11(51)
      brack(ninjaidxt0x1mu0)=acd11(52)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d11h12_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd11h12_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      vecA0(1:4) = - a0(0:3)
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
end module     p2_gg_httbar_d11h12l132_qp
