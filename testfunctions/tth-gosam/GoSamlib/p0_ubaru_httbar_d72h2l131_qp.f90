module     p0_ubaru_httbar_d72h2l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity2d72h2l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond_t, d => metric_tensor
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
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd72h2_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd72
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd72h2_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(46) :: acd72
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd72(1)=abb72(13)
      acd72(2)=dotproduct(k2,ninjaE3)
      acd72(3)=abb72(11)
      acd72(4)=dotproduct(l3,ninjaE3)
      acd72(5)=abb72(23)
      acd72(6)=dotproduct(l4,ninjaE3)
      acd72(7)=abb72(21)
      acd72(8)=dotproduct(ninjaA,ninjaE3)
      acd72(9)=dotproduct(ninjaE3,spval3k1)
      acd72(10)=abb72(10)
      acd72(11)=dotproduct(ninjaE3,spvak2l3)
      acd72(12)=abb72(12)
      acd72(13)=dotproduct(ninjaE3,spval3k2)
      acd72(14)=abb72(15)
      acd72(15)=dotproduct(ninjaE3,spval4l3)
      acd72(16)=abb72(16)
      acd72(17)=dotproduct(ninjaE3,spval4k2)
      acd72(18)=abb72(18)
      acd72(19)=dotproduct(ninjaE3,spval4k1)
      acd72(20)=abb72(19)
      acd72(21)=dotproduct(ninjaE3,spval3l4)
      acd72(22)=abb72(22)
      acd72(23)=dotproduct(k2,ninjaA)
      acd72(24)=dotproduct(l3,ninjaA)
      acd72(25)=dotproduct(l4,ninjaA)
      acd72(26)=dotproduct(ninjaA,ninjaA)
      acd72(27)=dotproduct(ninjaA,spval3k1)
      acd72(28)=dotproduct(ninjaA,spvak2l3)
      acd72(29)=dotproduct(ninjaA,spval3k2)
      acd72(30)=dotproduct(ninjaA,spval4l3)
      acd72(31)=dotproduct(ninjaA,spval4k2)
      acd72(32)=dotproduct(ninjaA,spval4k1)
      acd72(33)=dotproduct(ninjaA,spval3l4)
      acd72(34)=abb72(17)
      acd72(35)=acd72(2)*acd72(3)
      acd72(36)=acd72(4)*acd72(5)
      acd72(37)=acd72(6)*acd72(7)
      acd72(38)=acd72(8)*acd72(1)
      acd72(39)=acd72(9)*acd72(10)
      acd72(40)=acd72(11)*acd72(12)
      acd72(41)=acd72(13)*acd72(14)
      acd72(42)=acd72(15)*acd72(16)
      acd72(43)=acd72(17)*acd72(18)
      acd72(44)=acd72(19)*acd72(20)
      acd72(45)=acd72(21)*acd72(22)
      acd72(35)=acd72(45)+acd72(44)+acd72(43)+acd72(42)+acd72(41)+acd72(40)+acd&
      &72(39)+2.0_ki*acd72(38)+acd72(37)+acd72(35)+acd72(36)
      acd72(36)=ninjaP+acd72(26)
      acd72(36)=acd72(1)*acd72(36)
      acd72(37)=acd72(23)*acd72(3)
      acd72(38)=acd72(24)*acd72(5)
      acd72(39)=acd72(25)*acd72(7)
      acd72(40)=acd72(27)*acd72(10)
      acd72(41)=acd72(28)*acd72(12)
      acd72(42)=acd72(29)*acd72(14)
      acd72(43)=acd72(30)*acd72(16)
      acd72(44)=acd72(31)*acd72(18)
      acd72(45)=acd72(32)*acd72(20)
      acd72(46)=acd72(33)*acd72(22)
      acd72(36)=acd72(34)+acd72(46)+acd72(45)+acd72(44)+acd72(43)+acd72(42)+acd&
      &72(41)+acd72(40)+acd72(39)+acd72(37)+acd72(38)+acd72(36)
      brack(ninjaidxt1mu0)=acd72(35)
      brack(ninjaidxt0mu0)=acd72(36)
      brack(ninjaidxt0mu2)=acd72(1)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d72h2_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd72h2_qp
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
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p0_ubaru_httbar_d72h2l131_qp
