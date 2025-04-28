module     p0_ubaru_httbar_d22h13l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity13d22h13l1.f90
   ! generator: buildfortran.py
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_util, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd22h13
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc22(21)
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspl5
      complex(ki) :: Qspl3
      complex(ki) :: Qspk2
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspl5 = dotproduct(Q,l5)
      Qspl3 = dotproduct(Q,l3)
      Qspk2 = dotproduct(Q,k2)
      acc22(1)=abb22(12)
      acc22(2)=abb22(13)
      acc22(3)=abb22(14)
      acc22(4)=abb22(15)
      acc22(5)=abb22(16)
      acc22(6)=abb22(17)
      acc22(7)=abb22(18)
      acc22(8)=abb22(19)
      acc22(9)=abb22(21)
      acc22(10)=abb22(22)
      acc22(11)=abb22(35)
      acc22(12)=Qspval5l3*acc22(8)
      acc22(13)=Qspval3l5*acc22(9)
      acc22(14)=Qspval3k2*acc22(10)
      acc22(15)=Qspvak2l5*acc22(3)
      acc22(16)=Qspvak2l3*acc22(1)
      acc22(17)=Qspvak1l5*acc22(5)
      acc22(18)=Qspvak1l3*acc22(7)
      acc22(19)=Qspl5*acc22(11)
      acc22(20)=Qspl3*acc22(6)
      acc22(21)=Qspk2*acc22(2)
      brack=acc22(4)+acc22(12)+acc22(13)+acc22(14)+acc22(15)+acc22(16)+acc22(17&
      &)+acc22(18)+acc22(19)+acc22(20)+acc22(21)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d22h13l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd22h13
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d22
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      Q(1:4)  =cmplx(real(-Q_ext(0:3),  ki_nin), aimag(-Q_ext(0:3)), ki)
      d22 = 0.0_ki
      d22 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d22, ki), aimag(d22), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d22h13l1
