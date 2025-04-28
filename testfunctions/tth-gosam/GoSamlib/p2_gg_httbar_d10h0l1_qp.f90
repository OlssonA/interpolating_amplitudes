module     p2_gg_httbar_d10h0l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d10h0l1_qp.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd10h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc10(28)
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspk2
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak1l3
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspk2 = dotproduct(Q,k2)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      acc10(1)=abb10(9)
      acc10(2)=abb10(10)
      acc10(3)=abb10(11)
      acc10(4)=abb10(12)
      acc10(5)=abb10(13)
      acc10(6)=abb10(14)
      acc10(7)=abb10(15)
      acc10(8)=abb10(16)
      acc10(9)=abb10(17)
      acc10(10)=abb10(18)
      acc10(11)=abb10(19)
      acc10(12)=abb10(20)
      acc10(13)=abb10(21)
      acc10(14)=abb10(22)
      acc10(15)=abb10(25)
      acc10(16)=abb10(26)
      acc10(17)=abb10(27)
      acc10(18)=abb10(28)
      acc10(19)=acc10(4)*Qspval3k1
      acc10(20)=acc10(17)*Qspval5k1
      acc10(21)=acc10(18)*Qspval4k1
      acc10(19)=acc10(21)+acc10(20)+acc10(10)+acc10(19)
      acc10(19)=Qspvak1k2*acc10(19)
      acc10(20)=acc10(3)*Qspval3k2
      acc10(21)=-acc10(9)*Qspval5k2
      acc10(22)=-acc10(14)*Qspval4k2
      acc10(20)=acc10(22)+acc10(11)+acc10(21)+acc10(20)
      acc10(20)=Qspk2*acc10(20)
      acc10(21)=-acc10(15)*Qspval5k2
      acc10(21)=acc10(12)+acc10(21)
      acc10(21)=Qspvak2l3*acc10(21)
      acc10(22)=acc10(15)*Qspval5k1
      acc10(22)=acc10(16)+acc10(22)
      acc10(22)=Qspvak1l3*acc10(22)
      acc10(23)=acc10(1)*Qspval3k2
      acc10(24)=acc10(2)*Qspval4k1
      acc10(25)=acc10(5)*Qspval5k2
      acc10(26)=acc10(6)*Qspval3k1
      acc10(27)=acc10(8)*Qspval4k2
      acc10(28)=acc10(13)*Qspval5k1
      brack=acc10(7)+acc10(19)+acc10(20)+acc10(21)+acc10(22)+acc10(23)+acc10(24&
      &)+acc10(25)+acc10(26)+acc10(27)+acc10(28)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d10h0l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd10h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d10
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k5
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d10 = 0.0_ki
      d10 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d10, ki), aimag(d10), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d10h0l1_qp
