module     p2_gg_httbar_d177h12l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d177h12l132_qp.f90
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
      use p2_gg_httbar_abbrevd177h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(24) :: acd177
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd177(1)=dotproduct(ninjaE3,spval4e1)
      acd177(2)=dotproduct(ninjaE3,spvae1l4)
      acd177(3)=abb177(12)
      acd177(4)=dotproduct(ninjaE3,spvak1e1)
      acd177(5)=abb177(14)
      acd177(6)=dotproduct(ninjaE3,spvak2e1)
      acd177(7)=abb177(28)
      acd177(8)=dotproduct(ninjaE3,spval5e1)
      acd177(9)=abb177(20)
      acd177(10)=dotproduct(ninjaE3,spvae2e1)
      acd177(11)=abb177(40)
      acd177(12)=dotproduct(ninjaE3,spvae1k2)
      acd177(13)=abb177(19)
      acd177(14)=dotproduct(ninjaE3,spvae1e2)
      acd177(15)=abb177(26)
      acd177(16)=dotproduct(ninjaE3,spvae1l5)
      acd177(17)=abb177(27)
      acd177(18)=dotproduct(ninjaE3,spvae1k1)
      acd177(19)=abb177(44)
      acd177(20)=acd177(7)*acd177(2)
      acd177(21)=acd177(13)*acd177(12)
      acd177(22)=acd177(15)*acd177(14)
      acd177(23)=acd177(17)*acd177(16)
      acd177(24)=acd177(19)*acd177(18)
      acd177(20)=acd177(24)+acd177(23)+acd177(22)+acd177(21)+acd177(20)
      acd177(20)=acd177(6)*acd177(20)
      acd177(21)=acd177(3)*acd177(1)
      acd177(22)=acd177(5)*acd177(4)
      acd177(23)=-acd177(9)*acd177(8)
      acd177(24)=-acd177(11)*acd177(10)
      acd177(21)=acd177(24)+acd177(23)+acd177(21)+acd177(22)
      acd177(21)=acd177(2)*acd177(21)
      acd177(20)=acd177(20)+acd177(21)
      brack(ninjaidxt1x0mu0)=acd177(20)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd177h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(62) :: acd177
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd177(1)=dotproduct(ninjaA1,spval4e1)
      acd177(2)=dotproduct(ninjaE3,spvae1l4)
      acd177(3)=abb177(12)
      acd177(4)=dotproduct(ninjaA1,spvae1l4)
      acd177(5)=dotproduct(ninjaE3,spval4e1)
      acd177(6)=dotproduct(ninjaE3,spvak2e1)
      acd177(7)=abb177(28)
      acd177(8)=dotproduct(ninjaE3,spvak1e1)
      acd177(9)=abb177(14)
      acd177(10)=dotproduct(ninjaE3,spval5e1)
      acd177(11)=abb177(20)
      acd177(12)=dotproduct(ninjaE3,spvae2e1)
      acd177(13)=abb177(40)
      acd177(14)=dotproduct(ninjaA1,spvak2e1)
      acd177(15)=dotproduct(ninjaE3,spvae1k2)
      acd177(16)=abb177(19)
      acd177(17)=dotproduct(ninjaE3,spvae1k1)
      acd177(18)=abb177(44)
      acd177(19)=dotproduct(ninjaE3,spvae1e2)
      acd177(20)=abb177(26)
      acd177(21)=dotproduct(ninjaE3,spvae1l5)
      acd177(22)=abb177(27)
      acd177(23)=dotproduct(ninjaA1,spvak1e1)
      acd177(24)=dotproduct(ninjaA1,spvae1k2)
      acd177(25)=dotproduct(ninjaA1,spval5e1)
      acd177(26)=dotproduct(ninjaA1,spvae2e1)
      acd177(27)=dotproduct(ninjaA1,spvae1k1)
      acd177(28)=dotproduct(ninjaA1,spvae1e2)
      acd177(29)=dotproduct(ninjaA1,spvae1l5)
      acd177(30)=dotproduct(ninjaA0,spval4e1)
      acd177(31)=dotproduct(ninjaA0,spvae1l4)
      acd177(32)=dotproduct(ninjaA0,spvak2e1)
      acd177(33)=dotproduct(ninjaA0,spvak1e1)
      acd177(34)=dotproduct(ninjaA0,spvae1k2)
      acd177(35)=dotproduct(ninjaA0,spval5e1)
      acd177(36)=dotproduct(ninjaA0,spvae2e1)
      acd177(37)=dotproduct(ninjaA0,spvae1k1)
      acd177(38)=dotproduct(ninjaA0,spvae1e2)
      acd177(39)=dotproduct(ninjaA0,spvae1l5)
      acd177(40)=abb177(46)
      acd177(41)=abb177(15)
      acd177(42)=abb177(13)
      acd177(43)=abb177(24)
      acd177(44)=abb177(23)
      acd177(45)=abb177(65)
      acd177(46)=abb177(21)
      acd177(47)=abb177(22)
      acd177(48)=abb177(52)
      acd177(49)=abb177(59)
      acd177(50)=acd177(22)*acd177(29)
      acd177(51)=acd177(20)*acd177(28)
      acd177(52)=acd177(18)*acd177(27)
      acd177(53)=acd177(16)*acd177(24)
      acd177(54)=acd177(4)*acd177(7)
      acd177(50)=acd177(54)+acd177(53)+acd177(52)+acd177(50)+acd177(51)
      acd177(50)=acd177(6)*acd177(50)
      acd177(51)=-acd177(13)*acd177(26)
      acd177(52)=-acd177(11)*acd177(25)
      acd177(53)=acd177(9)*acd177(23)
      acd177(54)=acd177(3)*acd177(1)
      acd177(55)=acd177(14)*acd177(7)
      acd177(51)=acd177(55)+acd177(54)+acd177(53)+acd177(51)+acd177(52)
      acd177(51)=acd177(2)*acd177(51)
      acd177(52)=acd177(22)*acd177(21)
      acd177(53)=acd177(20)*acd177(19)
      acd177(54)=acd177(18)*acd177(17)
      acd177(55)=acd177(16)*acd177(15)
      acd177(52)=acd177(52)+acd177(53)+acd177(54)+acd177(55)
      acd177(53)=acd177(14)*acd177(52)
      acd177(54)=acd177(13)*acd177(12)
      acd177(55)=acd177(11)*acd177(10)
      acd177(56)=acd177(9)*acd177(8)
      acd177(57)=acd177(3)*acd177(5)
      acd177(54)=-acd177(54)-acd177(55)+acd177(56)+acd177(57)
      acd177(55)=acd177(4)*acd177(54)
      acd177(50)=acd177(51)+acd177(50)+acd177(53)+acd177(55)
      acd177(51)=acd177(22)*acd177(39)
      acd177(53)=acd177(20)*acd177(38)
      acd177(55)=acd177(18)*acd177(37)
      acd177(56)=acd177(16)*acd177(34)
      acd177(57)=acd177(31)*acd177(7)
      acd177(51)=acd177(57)+acd177(56)+acd177(55)+acd177(53)+acd177(42)+acd177(&
      &51)
      acd177(51)=acd177(6)*acd177(51)
      acd177(53)=-acd177(13)*acd177(36)
      acd177(55)=-acd177(11)*acd177(35)
      acd177(56)=acd177(9)*acd177(33)
      acd177(57)=acd177(3)*acd177(30)
      acd177(58)=acd177(32)*acd177(7)
      acd177(53)=acd177(58)+acd177(57)+acd177(56)+acd177(55)+acd177(41)+acd177(&
      &53)
      acd177(53)=acd177(2)*acd177(53)
      acd177(52)=acd177(32)*acd177(52)
      acd177(54)=acd177(31)*acd177(54)
      acd177(55)=acd177(21)*acd177(49)
      acd177(56)=acd177(19)*acd177(48)
      acd177(57)=acd177(17)*acd177(47)
      acd177(58)=acd177(15)*acd177(44)
      acd177(59)=acd177(12)*acd177(46)
      acd177(60)=acd177(10)*acd177(45)
      acd177(61)=acd177(8)*acd177(43)
      acd177(62)=acd177(5)*acd177(40)
      acd177(51)=acd177(53)+acd177(51)+acd177(54)+acd177(52)+acd177(62)+acd177(&
      &61)+acd177(60)+acd177(59)+acd177(58)+acd177(57)+acd177(55)+acd177(56)
      brack(ninjaidxt0x0mu0)=acd177(51)
      brack(ninjaidxt0x1mu0)=acd177(50)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d177h12_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd177h12_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k4
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
end module     p2_gg_httbar_d177h12l132_qp
